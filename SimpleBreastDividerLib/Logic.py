import os
import sys
import shutil
import tempfile
import logging

import vtk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

class SimpleBreastDividerLogic(ScriptedLoadableModuleLogic):
    """
    SimpleBreastDividerLogic (算法与业务逻辑层)
    负责核心的医学图像处理、3D 网格生成、形态学计算，以及调度独立的 nnUNet 推理子进程。
    与 UI 层完全解耦，可独立用于自动化批处理脚本。
    """

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def calculate_eroded_volume(self, maskNode, target_erosion_mm):
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        # 1. 数据转换：确保从 Segmentation 或 LabelMap 都能读到数组
        if maskNode.IsA("vtkMRMLSegmentationNode"):
            tempLabelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(maskNode, tempLabelNode)
            mask_array = slicer.util.arrayFromVolume(tempLabelNode)
            spacing = tempLabelNode.GetSpacing()
            readNode = tempLabelNode
        else:
            mask_array = slicer.util.arrayFromVolume(maskNode)
            spacing = maskNode.GetSpacing()
            readNode = maskNode

        spacing_zyx = (spacing[2], spacing[1], spacing[0])
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        
        # 初始化结果
        results = {"Left": 0.0, "Right": 0.0}
        total_eroded_array = np.zeros_like(mask_array, dtype=np.uint8)

        # --- 核心逻辑：固定 2=Left, 1=Right ---
        mapping = {1: "Right", 2: "Left"}

        for label_val, side in mapping.items():
            single_mask = (mask_array == label_val)
            if not np.any(single_mask):
                continue
            
            # 执行收缩计算
            edt_map = distance_transform_edt(single_mask, sampling=spacing_zyx)
            eroded_mask = edt_map > target_erosion_mm
            
            # 计算该侧体积 (ml)
            vol_ml = (np.sum(eroded_mask) * voxel_vol) / 1000.0
            results[side] = vol_ml
            
            # 将收缩后的结果存入展示数组
            total_eroded_array[eroded_mask] = label_val

        # ==========================================
        # 2. 可视化：创建红色轮廓并应用 3D 渲染设置
        # ==========================================
        # 接收生成的节点
        erodedNode = self.display_eroded_outline(total_eroded_array, readNode, target_erosion_mm)

        # 应用朋友的 3D 可视化代码 (收缩线透明度设为 0.8)
        if erodedNode and erodedNode.IsA("vtkMRMLSegmentationNode"):
            erodedNode.CreateClosedSurfaceRepresentation()
            displayNode = erodedNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility3D(True)
                displayNode.SetOpacity3D(0.8)

        # 3. 清理临时节点
        if maskNode.IsA("vtkMRMLSegmentationNode"):
            slicer.mrmlScene.RemoveNode(tempLabelNode)

        return results["Left"], results["Right"]

    def display_eroded_outline(self, array, refNode, mm):
        """将收缩后的结果显示为红色轮廓（修复了颜色设置报错问题）"""
        new_name = f"Eroded_{mm}mm_Outline"
        
        # 清理同名旧节点
        old = slicer.mrmlScene.GetFirstNodeByName(new_name)
        if old:
            slicer.mrmlScene.RemoveNode(old)

        # 创建新的分割节点
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", new_name)
        
        # 将数组转为临时 Labelmap 再导入
        tempLabel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        tempLabel.CopyOrientation(refNode)
        slicer.util.updateVolumeFromArray(tempLabel, array)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tempLabel, segNode)
        slicer.mrmlScene.RemoveNode(tempLabel)

        # 获取显示和逻辑对象
        displayNode = segNode.GetDisplayNode()
        segmentation = segNode.GetSegmentation()

        # 遍历所有 Segment，设置外观
        for i in range(segmentation.GetNumberOfSegments()):
            s_id = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(s_id)
            
            # 1. 设置颜色为纯红 (R=1, G=0, B=0) - 直接修改 Segment 对象，最稳健
            segment.SetColor(1, 0, 0)
            
            # 2. 修改显示样式
            if displayNode:
                # 关闭 2D 填充，开启轮廓
                displayNode.SetSegmentOpacity2DFill(s_id, 0)
                displayNode.SetSegmentOpacity2DOutline(s_id, 1.0)
                # 设置 3D 透明度
                displayNode.SetSegmentOpacity3D(s_id, 0.4)

        return segNode

    def _arrayToLabelmap(self, array, referenceNode):
        """辅助函数：将数组转回 Labelmap 以便导入 Segmentation"""
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        labelNode.CopyOrientation(referenceNode)
        slicer.util.updateVolumeFromArray(labelNode, array)
        return labelNode

    def extract_gland_volume(self, imageNode, maskNode):
        import numpy as np
        from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, generate_binary_structure
        
        # --- 自动处理 sklearn 依赖 ---
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            slicer.util.showStatusMessage("正在自动安装 scikit-learn，这只需几秒钟...")
            slicer.app.processEvents()
            slicer.util.pip_install("scikit-learn -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com")
            from sklearn.cluster import KMeans

        # 1. 提取图像 Numpy 数组
        image_array = slicer.util.arrayFromVolume(imageNode)
        
        # === 核心修复区：强制空间与尺寸对齐 ===
        if maskNode.IsA("vtkMRMLSegmentationNode"):
            tempLabelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(maskNode, tempLabelNode, imageNode)
            mask_array = slicer.util.arrayFromVolume(tempLabelNode)
            readNode = tempLabelNode
        else:
            mask_array = slicer.util.arrayFromVolume(maskNode)
            if mask_array.shape != image_array.shape:
                raise ValueError(f"尺寸不匹配！原图形状 {image_array.shape}，但 Mask 形状 {mask_array.shape}。请选择正确的对应影像。")
            readNode = maskNode

        spacing = imageNode.GetSpacing()
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        
        blur_sigma_pre = 0.6
        morph_iter = 2
        rounding_sigma = 0.8
        struct_element = generate_binary_structure(rank=3, connectivity=1)

        data_smooth = gaussian_filter(image_array, sigma=blur_sigma_pre)
        
        results = {"Left": 0.0, "Right": 0.0}
        total_gland_array = np.zeros_like(mask_array, dtype=np.uint8)
        mapping = {1: "Right", 2: "Left"}

        # 2. 分别处理左右乳房
        for label_val, side in mapping.items():
            mask_indices = (mask_array == label_val)
            if not np.any(mask_indices):
                continue
            
            valid_pixels = data_smooth[mask_indices]
            if len(valid_pixels) == 0: 
                continue

            X = valid_pixels.reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            
            target_label_index = np.argmax(centers)
            
            seg_temp = np.zeros_like(image_array, dtype=np.uint8)
            seg_temp[mask_indices] = (labels == target_label_index).astype(np.uint8)
            
            seg_closed = binary_closing(seg_temp, structure=struct_element, iterations=morph_iter)
            seg_morph_done = binary_opening(seg_closed, structure=struct_element, iterations=1)
            
            seg_float = seg_morph_done.astype(np.float32)
            seg_blurred = gaussian_filter(seg_float, sigma=rounding_sigma)
            seg_final = (seg_blurred > 0.5).astype(np.uint8)
            
            results[side] = (np.sum(seg_final) * voxel_vol) / 1000.0
            total_gland_array[seg_final > 0] = label_val

        # ==========================================
        # 3. 可视化：生成蓝色实体掩膜并应用 3D 渲染设置
        # ==========================================
        # 接收生成的节点
        glandNode = self.display_gland_outline(total_gland_array, readNode) 

        # 应用朋友的 3D 可视化代码 (腺体透明度设为 0.5)
        if glandNode and glandNode.IsA("vtkMRMLSegmentationNode"):
            glandNode.CreateClosedSurfaceRepresentation()
            displayNode = glandNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility3D(True)
                displayNode.SetOpacity3D(0.5)

        if maskNode.IsA("vtkMRMLSegmentationNode"):
            slicer.mrmlScene.RemoveNode(tempLabelNode)

        return results["Left"], results["Right"]

    def display_gland_outline(self, array, refNode):
        """将提取的腺体显示为蓝色轮廓线（无填充蒙版）"""
        new_name = "Glands_KMeans_Extracted"
        old = slicer.mrmlScene.GetFirstNodeByName(new_name)
        if old: slicer.mrmlScene.RemoveNode(old)

        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", new_name)
        tempLabel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        tempLabel.CopyOrientation(refNode)
        slicer.util.updateVolumeFromArray(tempLabel, array)
        
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tempLabel, segNode)
        slicer.mrmlScene.RemoveNode(tempLabel)

        displayNode = segNode.GetDisplayNode()
        segmentation = segNode.GetSegmentation()

        for i in range(segmentation.GetNumberOfSegments()):
            s_id = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(s_id)
            # 设置腺体为亮蓝色 (RGB: 0, 0.8, 1)
            segment.SetColor(0.0, 0.8, 1.0)
            
            if displayNode:
                # --- 核心修改区 ---
                displayNode.SetSegmentOpacity2DFill(s_id, 0.0)     # 【设为0】完全关闭 2D 填充蒙版
                displayNode.SetSegmentOpacity2DOutline(s_id, 1.0)  # 【设为1】保持 100% 蓝色轮廓线
                displayNode.SetSegmentOpacity3D(s_id, 0.3)         # 3D 视图中调淡，避免喧宾夺主

        return segNode

    def run(self, inputVolume, modelPath, name_prefix="Output", color=None, constrainMaskNode=None):
        import tempfile
        import shutil
        import os
        import subprocess
        import sys

        slicer.util.showStatusMessage(f"正在准备执行 {name_prefix} 分割...")
        slicer.app.processEvents()

        # 1. 创建临时文件夹
        tempDir = tempfile.mkdtemp(prefix="BreastDivider_")
        inputDir = os.path.join(tempDir, "input")
        outputDir = os.path.join(tempDir, "output")
        os.makedirs(inputDir)
        os.makedirs(outputDir)

        try:
            # 2. 导出数据 (新增强制成功校验)
            case_id = "testcase"
            inputFilePath = os.path.join(inputDir, f"{case_id}_0000.nii.gz")
            success = slicer.util.saveNode(inputVolume, inputFilePath)
            if not success or not os.path.exists(inputFilePath):
                raise RuntimeError(f"Slicer 无法将影像导出到临时目录！请检查影像是否正常。")

            slicer.util.showStatusMessage("后台推理中，请耐心等待 (Slicer可能短暂无响应)...")
            slicer.app.processEvents() 

            # 【核心修复】：统一转正斜杠，防止 Windows 路径转义 bug 吞噬文件夹名
            modelPath = modelPath.replace('\\', '/')
            inputDir = inputDir.replace('\\', '/')
            outputDir = outputDir.replace('\\', '/')
            
            # 后台动态脚本
            script_code = f"""
import torch
import multiprocessing
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    print("--- [DEBUG] STARTING DIRECT INFERENCE ---")
    print(r"Target Model Path: {modelPath}")
    print(r"Input Dir: {inputDir}")

    if torch.cuda.is_available():
        my_device = torch.device('cuda', 0)
        print("GPU Detected! Using CUDA.")
    else:
        my_device = torch.device('cpu')
        print("WARNING: Falling back to CPU.")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=my_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    print("Initializing model...")
    predictor.initialize_from_trained_model_folder(
        r'{modelPath}',
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth'
    )

    print("Starting prediction...")
    predictor.predict_from_files(
        r'{inputDir}',
        r'{outputDir}',
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=1, num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
    )
    print("Prediction completed successfully!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
"""
            script_path = os.path.join(tempDir, "run_inference.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_code)

            if sys.platform == "win32":
                python_executable = os.path.join(slicer.app.slicerHome, "bin", "PythonSlicer.exe")
            else:
                python_executable = os.path.join(slicer.app.slicerHome, "bin", "PythonSlicer")
                
            command = [python_executable, script_path]
            
            process = subprocess.run(command, capture_output=True, text=True)

            # 3. 结果校验与【日志强制捕捉】
            if process.returncode != 0:
                print("=== nnUNet 崩溃日志 ===")
                print(process.stdout)
                print(process.stderr)
                raise RuntimeError("nnUNet 引擎崩溃。请按 Ctrl+3 打开控制台查看红色日志详情。")

            outputFilePath = os.path.join(outputDir, f"{case_id}.nii.gz")
            if not os.path.exists(outputFilePath):
                print("=== nnUNet 幽灵运行日志 ===")
                print(process.stdout)
                print(process.stderr)
                raise FileNotFoundError("底层推理完成但未生成结果！可能是路径问题或模型未执行。请按 Ctrl+3 查看控制台诊断！")

            # === 修改 4. 加载结果后的逻辑 ===
            slicer.util.showStatusMessage("推理完成，正在执行空间约束过滤...")
            
            # 加载预测出的 LabelVolume (临时)
            segmentationLabelNode = slicer.util.loadLabelVolume(outputFilePath)
            pred_array = slicer.util.arrayFromVolume(segmentationLabelNode)

            # --- 核心：执行空间约束 (Masking) ---
            if constrainMaskNode:
                # 将约束 Mask 转为相同尺寸的 Numpy 数组
                tempLabel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                    constrainMaskNode, tempLabel, segmentationLabelNode) # 强制尺寸对齐
                constraint_array = slicer.util.arrayFromVolume(tempLabel)
                
                # 【像素级过滤】：病灶像素 * (乳腺掩膜 > 0)
                # 只有在乳腺范围内的病灶会被保留
                pred_array = pred_array * (constraint_array > 0)
                
                # 更新节点数据并清理
                slicer.util.updateVolumeFromArray(segmentationLabelNode, pred_array)
                slicer.mrmlScene.RemoveNode(tempLabel)
                print(f"[INFO] 已成功将 {name_prefix} 限制在 {constrainMaskNode.GetName()} 范围内")

            # --- 后续加载为 Segmentation 节点并上色 (保持之前代码即可) ---
            final_node_name = f"{name_prefix}_{inputVolume.GetName()}"
            finalSegNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", final_node_name)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segmentationLabelNode, finalSegNode)
            slicer.mrmlScene.RemoveNode(segmentationLabelNode) 
            
            # 【核心修复】：把这俩变量的定义拿出来，放在最外面！不管走哪个分支都能用到。
            displayNode = finalSegNode.GetDisplayNode()
            segmentation = finalSegNode.GetSegmentation()

            if color:
                # 场景 A: 病灶分割 (传入了特定的洋红色)
                for i in range(segmentation.GetNumberOfSegments()):
                    s_id = segmentation.GetNthSegmentID(i)
                    segment = segmentation.GetSegment(s_id)
                    
                    segment.SetColor(color[0], color[1], color[2])
                    if displayNode:
                        displayNode.SetSegmentOpacity2DFill(s_id, 0.6)
                        displayNode.SetSegmentOpacity3D(s_id, 0.7)
            else:
                # 场景 B: 乳房分割 (恢复绿/橙双色)
                for i in range(segmentation.GetNumberOfSegments()):
                    s_id = segmentation.GetNthSegmentID(i)
                    segment = segmentation.GetSegment(s_id)
                    
                    # 获取最准确的原始像素标签值
                    # --- 核心修复：创建一个 vtk.mutable 空盒子来接收 C++ 传回来的值 ---
                    labelValueMutable = vtk.mutable("")
                    segment.GetTag("Segmentation.LabelmapLabelValue", labelValueMutable)
                    
                    # 把盒子里的内容转成 Python 认识的普通字符串
                    labelValue = str(labelValueMutable)
                    
                    if labelValue == "2":
                        segment.SetColor(0.2, 0.8, 0.2) # 翠绿色 (左侧)
                    elif labelValue == "1":
                        segment.SetColor(1.0, 0.5, 0.0) # 纯橙色 (右侧)
                    else:
                        # 兜底匹配：万一 Tag 没取到，我们直接去读它在 Slicer 列表里显示的名字
                        segName = segment.GetName()
                        if "2" in segName:
                            segment.SetColor(0.2, 0.8, 0.2)
                        elif "1" in segName:
                            segment.SetColor(1.0, 0.5, 0.0)

                    if displayNode:
                        displayNode.SetSegmentOpacity2DFill(s_id, 0.25)
                        displayNode.SetSegmentOpacity3D(s_id, 0.35)

            # ====================================================
            # 🚀 新增功能：自动生成 3D 模型并居中视角
            # ====================================================
            slicer.util.showStatusMessage("正在生成 3D 表面模型...")
            
            # 1. 强制 Slicer 为这个掩膜生成封闭的 3D 表面 (Closed Surface)
            finalSegNode.CreateClosedSurfaceRepresentation()
            
            # 2. 自动调整 3D 窗口的摄像机，使其完美居中显示我们的模型
            layoutManager = slicer.app.layoutManager()
            if layoutManager:
                threeDWidget = layoutManager.threeDWidget(0)
                if threeDWidget:
                    threeDView = threeDWidget.threeDView()
                    # 重置焦点，确保模型完整出现在 3D 视口中央
                    threeDView.resetFocalPoint()
                    
            slicer.util.showStatusMessage("分割与 3D 渲染完成！", 3000)
            
            return finalSegNode

        except Exception as e:
            slicer.util.errorDisplay(f"{name_prefix} 运行出错: {str(e)}")
        finally:
            shutil.rmtree(tempDir, ignore_errors=True)
            slicer.util.showStatusMessage(f"{name_prefix} 模块运行结束！")

    
    def _getUnifiedPolyData(self, segNode):
        """辅助函数：将分割节点中可能存在的多个 Segment 合并提取为一个完整的 vtkPolyData 表面网格"""
        import vtk
        appendFilter = vtk.vtkAppendPolyData()
        segmentation = segNode.GetSegmentation()
        segNode.CreateClosedSurfaceRepresentation()
        
        added = False
        for i in range(segmentation.GetNumberOfSegments()):
            s_id = segmentation.GetNthSegmentID(i)
            poly = vtk.vtkPolyData()
            slicer.modules.segmentations.logic().GetSegmentClosedSurfaceRepresentation(segNode, s_id, poly)
            if poly.GetNumberOfPoints() > 0:
                appendFilter.AddInputData(poly)
                added = True
                
        if added:
            appendFilter.Update()
            return appendFilter.GetOutput()
        return None

    # def generate_distance_heatmap(self, breastNode, glandNode):
    #     import vtk
    #     import colorsys
        
    #     # 1. 提取网格 (保持之前逻辑)
    #     breastPoly = self._getUnifiedPolyData(breastNode)
    #     glandPoly = self._getUnifiedPolyData(glandNode)
    #     if not breastPoly or not glandPoly: raise ValueError("无法提取 3D 网格！")

    #     # 2. 计算距离与法线 (确保不发黑)
    #     distFilter = vtk.vtkDistancePolyDataFilter()
    #     distFilter.SetInputData(0, glandPoly)  
    #     distFilter.SetInputData(1, breastPoly) 
    #     distFilter.SignedDistanceOff()         
    #     distFilter.Update()

    #     normalsFilter = vtk.vtkPolyDataNormals()
    #     normalsFilter.SetInputData(distFilter.GetOutput())
    #     normalsFilter.ComputePointNormalsOn()
    #     normalsFilter.SplittingOff()
    #     normalsFilter.Update()
    #     heatmapPoly = normalsFilter.GetOutput()

    #     # 3. 获取距离范围
    #     heatmapPoly.GetPointData().SetActiveScalars("Distance")
    #     distance_array = heatmapPoly.GetPointData().GetArray("Distance")
    #     dist_min, dist_max = distance_array.GetRange() if distance_array else (0.0, 100.0)
    #     if dist_max == dist_min: dist_max = dist_min + 1.0

    #     # 4. 创建/更新模型节点
    #     model_name = "Gland_Distance_Heatmap_Model"
    #     oldNode = slicer.mrmlScene.GetFirstNodeByName(model_name)
    #     if oldNode: slicer.mrmlScene.RemoveNode(oldNode)
    #     modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", model_name)
    #     modelNode.SetAndObservePolyData(heatmapPoly)
    #     modelNode.CreateDefaultDisplayNodes()
    #     displayNode = modelNode.GetDisplayNode()

    #     # 5. 配置连续颜色映射 (Procedural Color Node)
    #     colorNodeName = "DistanceHeatmapColor_Continuous"
    #     colorNode = slicer.mrmlScene.GetFirstNodeByName(colorNodeName)
    #     if not colorNode:
    #         colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLProceduralColorNode", colorNodeName)
        
    #     colorTransferFunction = vtk.vtkColorTransferFunction()
    #     colorTransferFunction.AddRGBPoint(dist_min, 1.0, 0.0, 0.0) # 红
    #     colorTransferFunction.AddRGBPoint(dist_min + (dist_max - dist_min) * 0.5, 0.0, 1.0, 0.0) # 绿
    #     colorTransferFunction.AddRGBPoint(dist_max, 0.0, 0.0, 1.0) # 蓝
    #     colorNode.SetAndObserveColorTransferFunction(colorTransferFunction)

    #     # 6. 设置渲染属性
    #     displayNode.SetScalarVisibility(True)
    #     displayNode.SetActiveScalarName("Distance") 
    #     displayNode.SetAndObserveColorNodeID(colorNode.GetID())
    #     displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange) 
    #     displayNode.SetScalarRange(dist_min, dist_max) 
    #     displayNode.SetInterpolation(2)

    #     # ==========================================
    #     # === 配置 3D 颜色数轴 (Color Legend) ===
    #     # ==========================================
    #     colorLogic = slicer.modules.colors.logic()
    #     clNode = colorLogic.AddDefaultColorLegendDisplayNode(displayNode)
        
    #     if clNode:
    #         clNode.SetName("Gland_Distance_Heatmap_Legend") 
    #         clNode.SetTitleText("安全边界距离 (mm)")          
    #         clNode.SetLabelFormat("%4.1f")
            
    #         # === 【新增功能】：调整图示到左侧，并略微缩小 ===
    #         # X=0.02 (距离左侧边缘 2%), Y=0.15 (距离底部 15%)
    #         clNode.SetPosition(0.02, 0.15) 
    #         # 缩小宽度和高度 (Slicer 默认大约是 0.15 和 0.5)
    #         clNode.SetSize(0.1, 0.4)       

    #         try:
    #             clNode.SetNumberOfLabels(5)                 
    #         except AttributeError:
    #             pass 
                
    #         clNode.SetVisibility(True)

    def processDistanceHeatmap(self, targetNode, refNode):
        import vtk
        import colorsys
        import slicer
        import logging

        logging.info(f"开始计算空间测距: 目标层 [{targetNode.GetName()}] -> 参照层 [{refNode.GetName()}]")

        # 1. 提取网格 (完全保留你的统一提取逻辑)
        targetPoly = self._getUnifiedPolyData(targetNode)
        refPoly = self._getUnifiedPolyData(refNode)
        if not targetPoly or not refPoly: 
            raise ValueError("无法提取 3D 网格！请检查输入的节点。")

        # 2. 计算距离与法线 (保留你优秀的法线修复与绝对值计算)
        distFilter = vtk.vtkDistancePolyDataFilter()
        distFilter.SetInputData(0, targetPoly)  # 目标层：被画上热力图的模型
        distFilter.SetInputData(1, refPoly)     # 参照层：测距的基准模型
        distFilter.SignedDistanceOff()          # 强制绝对值距离
        distFilter.Update()

        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(distFilter.GetOutput())
        normalsFilter.ComputePointNormalsOn()
        normalsFilter.SplittingOff()
        normalsFilter.Update()
        heatmapPoly = normalsFilter.GetOutput()

        # 3. 获取距离范围
        heatmapPoly.GetPointData().SetActiveScalars("Distance")
        distance_array = heatmapPoly.GetPointData().GetArray("Distance")
        dist_min, dist_max = distance_array.GetRange() if distance_array else (0.0, 100.0)
        if dist_max == dist_min: dist_max = dist_min + 1.0

        # 4. 创建/更新模型节点 (去硬编码，使用动态名称，防止不同目标的模型互相覆盖)
        model_name = f"{targetNode.GetName()}_to_{refNode.GetName()}_Heatmap"
        oldNode = slicer.mrmlScene.GetFirstNodeByName(model_name)
        if oldNode: slicer.mrmlScene.RemoveNode(oldNode)
        
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", model_name)
        modelNode.SetAndObservePolyData(heatmapPoly)
        modelNode.CreateDefaultDisplayNodes()
        displayNode = modelNode.GetDisplayNode()

        # 5. 配置连续颜色映射 (保留你手写的自定义色带: 红->绿->蓝)
        # 注意：这里也需要动态命名，防止冲突
        colorNodeName = f"HeatmapColor_{model_name}"
        colorNode = slicer.mrmlScene.GetFirstNodeByName(colorNodeName)
        if not colorNode:
            colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLProceduralColorNode", colorNodeName)
        
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(dist_min, 1.0, 0.0, 0.0) # 红 (距离最近)
        colorTransferFunction.AddRGBPoint(dist_min + (dist_max - dist_min) * 0.5, 0.0, 1.0, 0.0) # 绿
        colorTransferFunction.AddRGBPoint(dist_max, 0.0, 0.0, 1.0) # 蓝 (距离最远)
        colorNode.SetAndObserveColorTransferFunction(colorTransferFunction)

        # 6. 设置渲染属性
        displayNode.SetScalarVisibility(True)
        displayNode.SetActiveScalarName("Distance") 
        displayNode.SetAndObserveColorNodeID(colorNode.GetID())
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange) 
        displayNode.SetScalarRange(dist_min, dist_max) 
        displayNode.SetInterpolation(2)

        # ==========================================
        # === 配置 3D 颜色数轴 (保留你完美的 UI 布局) ===
        # ==========================================
        colorLogic = slicer.modules.colors.logic()
        clNode = colorLogic.AddDefaultColorLegendDisplayNode(displayNode)
        
        if clNode:
            clNode.SetName(f"{model_name}_Legend") 
            clNode.SetTitleText("安全边界距离 (mm)")          
            clNode.SetLabelFormat("%4.1f")
            clNode.SetPosition(0.02, 0.15) 
            clNode.SetSize(0.1, 0.4)       

            try:
                clNode.SetNumberOfLabels(5)                 
            except AttributeError:
                pass 
                
            clNode.SetVisibility(True)
