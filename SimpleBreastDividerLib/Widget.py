import logging
import os
import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin

# 通过相对导入引入同包下的 Logic 逻辑控制类
from .Logic import SimpleBreastDividerLogic

class SimpleBreastDividerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    SimpleBreastDividerWidget (UI与交互层)
    负责 3D Slicer 侧边栏面板的 Qt 控件渲染、排版以及用户点击事件的绑定调度。
    """

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # =========================================================================
        # === 模块 1：基础设置与 AI 分割执行区 ===
        # =========================================================================
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "分割模型配置与执行"
        self.layout.addWidget(parametersCollapsibleButton)
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # --- 采用方案一：相对路径自动推导 ---
        
        # 1. 获取当前 Widget.py 文件所在的绝对路径 
        # (此时路径在: .../BreastDividerTool/BreastLesionAnalyzer/BreastLesionAnalyzerLib)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 向上推两级，精确定位到 BreastDividerTool 根目录
        # 往上一级: BreastLesionAnalyzer 插件目录
        # 再往上一级: BreastDividerTool 总目录
        root_dir = os.path.dirname(os.path.dirname(current_dir))

        # 3. 动态拼装模型路径 (自动适应不同的电脑和盘符)
        default_breast_model = os.path.join(root_dir, "Dataset001_Breast", "nnUNetTrainer__nnUNetPlans__3d_fullres")
        default_lesion_model = os.path.join(root_dir, "Dataset102_BreastLesion", "nnUNetTrainer__nnUNetPlans__3d_fullres")
        
        # 4. 统一转换斜杠，防止 Windows 路径转义引发的 Bug
        default_breast_model = default_breast_model.replace('\\', '/')
        default_lesion_model = default_lesion_model.replace('\\', '/')
        # ----------------------------------

        # 1. 乳房区域分割模型路径
        self.breastModelSelector = ctk.ctkPathLineEdit()
        self.breastModelSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.breastModelSelector.currentPath = default_breast_model
        parametersFormLayout.addRow("乳房区域分割模型路径: ", self.breastModelSelector)

        # 2. 病灶区域分割模型路径
        self.lesionModelSelector = ctk.ctkPathLineEdit()
        self.lesionModelSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.lesionModelSelector.currentPath = default_lesion_model
        parametersFormLayout.addRow("病灶区域分割模型路径: ", self.lesionModelSelector)

        # 3. 输入影像选择
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.toolTip = "选择要预测的输入影像"
        parametersFormLayout.addRow("输入影像: ", self.inputSelector)

        # 4. 执行乳房分割按钮 (黄色)
        self.applyBreastButton = qt.QPushButton("执行乳房区域分割")
        self.applyBreastButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #FFC107; color: white;")
        parametersFormLayout.addRow(self.applyBreastButton)

        # 空间约束掩膜
        self.lesionConstraintSelector = slicer.qMRMLNodeComboBox()
        self.lesionConstraintSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode"]
        self.lesionConstraintSelector.addEnabled = False
        self.lesionConstraintSelector.renameEnabled = True
        self.lesionConstraintSelector.setMRMLScene(slicer.mrmlScene)
        self.lesionConstraintSelector.toolTip = "选择乳腺掩膜，将病灶分割限制在其范围内"
        parametersFormLayout.addRow("空间约束掩膜 (乳腺): ", self.lesionConstraintSelector)
        
        # 5. 执行病灶分割按钮 (红色警示)
        self.applyLesionButton = qt.QPushButton("执行病灶区域分割")
        self.applyLesionButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #F44336; color: white;")
        parametersFormLayout.addRow(self.applyLesionButton)

        # 绑定点击事件
        self.applyBreastButton.connect('clicked(bool)', self.onApplyBreast)
        self.applyLesionButton.connect('clicked(bool)', self.onApplyLesion)

        # =========================================================================
        # === 模块 2：乳腺上限体积计算面板 ===
        # =========================================================================
        volumeCollapsibleButton = ctk.ctkCollapsibleButton()
        volumeCollapsibleButton.text = "乳腺体积分析 (收缩计算)"
        self.layout.addWidget(volumeCollapsibleButton)
        volumeFormLayout = qt.QFormLayout(volumeCollapsibleButton)

        # 选择要计算的分割结果
        self.maskSelector = slicer.qMRMLNodeComboBox()
        self.maskSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode"]
        self.maskSelector.selectNodeUponCreation = True
        self.maskSelector.addEnabled = False
        self.maskSelector.removeEnabled = False
        self.maskSelector.noneEnabled = False
        self.maskSelector.showHidden = False
        self.maskSelector.showChildNodeTypes = False
        self.maskSelector.setMRMLScene(slicer.mrmlScene)
        volumeFormLayout.addRow("选择分割结果: ", self.maskSelector)

        # 收缩距离设置 (默认 5.0 mm)
        self.erosionSpinBox = qt.QDoubleSpinBox()
        self.erosionSpinBox.setValue(5.0)
        self.erosionSpinBox.suffix = " mm"
        volumeFormLayout.addRow("边缘收缩距离: ", self.erosionSpinBox)

        # 计算体积按钮
        self.calcVolumeButton = qt.QPushButton("计算乳腺上限体积")
        self.calcVolumeButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #4CAF50; color: white;")
        volumeFormLayout.addRow(self.calcVolumeButton)

        # 体积结果输出显示框 (只读)
        self.leftResultTextBox = qt.QLineEdit()
        self.leftResultTextBox.setReadOnly(True)
        self.leftResultTextBox.text = "左侧: 0.00 ml"
        volumeFormLayout.addRow("左侧乳房上限体积: ", self.leftResultTextBox)

        self.rightResultTextBox = qt.QLineEdit()
        self.rightResultTextBox.setReadOnly(True)
        self.rightResultTextBox.text = "右侧: 0.00 ml"
        volumeFormLayout.addRow("右侧乳房上限体积: ", self.rightResultTextBox)
        
        # 绑定点击事件
        self.calcVolumeButton.connect('clicked(bool)', self.onCalcVolume)

        # =========================================================================
        # === 模块 3：乳腺腺体提取功能面板 (K-Means) ===
        # =========================================================================
        glandCollapsibleButton = ctk.ctkCollapsibleButton()
        glandCollapsibleButton.text = "乳腺腺体提取与体积计算 (K-Means)"
        self.layout.addWidget(glandCollapsibleButton)
        glandFormLayout = qt.QFormLayout(glandCollapsibleButton)

        # 选择原始影像
        self.glandImageSelector = slicer.qMRMLNodeComboBox()
        self.glandImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.glandImageSelector.selectNodeUponCreation = True
        self.glandImageSelector.addEnabled = False
        self.glandImageSelector.removeEnabled = False
        self.glandImageSelector.noneEnabled = False
        self.glandImageSelector.showHidden = False
        self.glandImageSelector.showChildNodeTypes = False
        self.glandImageSelector.setMRMLScene(slicer.mrmlScene)
        glandFormLayout.addRow("选择原始影像: ", self.glandImageSelector)

        # 选择限制范围的 Mask
        self.glandMaskSelector = slicer.qMRMLNodeComboBox()
        self.glandMaskSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode"]
        self.glandMaskSelector.selectNodeUponCreation = True
        self.glandMaskSelector.addEnabled = False
        self.glandMaskSelector.removeEnabled = False
        self.glandMaskSelector.noneEnabled = False
        self.glandMaskSelector.showHidden = False
        self.glandMaskSelector.showChildNodeTypes = False
        self.glandMaskSelector.setMRMLScene(slicer.mrmlScene)
        glandFormLayout.addRow("选择乳腺范围(Mask): ", self.glandMaskSelector)

        # 提取计算按钮
        self.calcGlandButton = qt.QPushButton("执行 K-Means 腺体提取")
        self.calcGlandButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #2196F3; color: white;")
        glandFormLayout.addRow(self.calcGlandButton)

        # 结果显示框
        self.leftGlandTextBox = qt.QLineEdit()
        self.leftGlandTextBox.setReadOnly(True)
        self.leftGlandTextBox.text = "左侧腺体: 0.00 ml"
        glandFormLayout.addRow("左侧腺体体积: ", self.leftGlandTextBox)

        self.rightGlandTextBox = qt.QLineEdit()
        self.rightGlandTextBox.setReadOnly(True)
        self.rightGlandTextBox.text = "右侧腺体: 0.00 ml"
        glandFormLayout.addRow("右侧腺体体积: ", self.rightGlandTextBox)

        # 绑定点击事件
        self.calcGlandButton.connect('clicked(bool)', self.onCalcGland)

        # =========================================================================
        # === 模块 4：安全边界距离热力图计算面板 ===
        # =========================================================================
        distanceCollapsibleButton = ctk.ctkCollapsibleButton()
        distanceCollapsibleButton.text = "腺体安全边界分析 (3D 热力图)"
        self.layout.addWidget(distanceCollapsibleButton)
        distanceFormLayout = qt.QFormLayout(distanceCollapsibleButton)

        # 选择最外层轮廓 (皮肤)
        self.distBreastSelector = slicer.qMRMLNodeComboBox()
        self.distBreastSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.distBreastSelector.selectNodeUponCreation = True
        self.distBreastSelector.addEnabled = False
        self.distBreastSelector.removeEnabled = False
        self.distBreastSelector.noneEnabled = False
        self.distBreastSelector.showHidden = False
        self.distBreastSelector.setMRMLScene(slicer.mrmlScene)
        distanceFormLayout.addRow("选择最外层轮廓(乳房): ", self.distBreastSelector)

        # 选择内部结构 (腺体)
        self.distGlandSelector = slicer.qMRMLNodeComboBox()
        self.distGlandSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.distGlandSelector.selectNodeUponCreation = True
        self.distGlandSelector.addEnabled = False
        self.distGlandSelector.removeEnabled = False
        self.distGlandSelector.noneEnabled = False
        self.distGlandSelector.showHidden = False
        self.distGlandSelector.setMRMLScene(slicer.mrmlScene)
        distanceFormLayout.addRow("选择内部结构(腺体): ", self.distGlandSelector)

        # 渲染热力图按钮
        self.calcDistanceButton = qt.QPushButton("生成 3D 距离热力图")
        self.calcDistanceButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #9C27B0; color: white;")
        distanceFormLayout.addRow(self.calcDistanceButton)
        
        # 绑定点击事件
        self.calcDistanceButton.connect('clicked(bool)', self.onCalcHeatmap)

        # =========================================================================
        # === 模块 5：可视化与 3D 视图层级管理面板 ===
        # =========================================================================
        self.displayCollapsibleButton = ctk.ctkCollapsibleButton()
        self.displayCollapsibleButton.text = "3D 视图层级 management"
        self.layout.addWidget(self.displayCollapsibleButton)
        displayLayout = qt.QFormLayout(self.displayCollapsibleButton)

        def create_visibility_row(label_text, default_opacity):
            checkbox = qt.QCheckBox(label_text)
            checkbox.checked = True
            slider = ctk.ctkSliderWidget()
            slider.singleStep = 0.1
            slider.minimum = 0.0
            slider.maximum = 1.0
            slider.value = default_opacity 
            slider.decimals = 2
            return checkbox, slider

        self.showBreastCheck, self.breastOpacitySlider = create_visibility_row("乳房整体区域 (绿/橙)", 0.35)
        self.showErodedCheck, self.erodedOpacitySlider = create_visibility_row("边缘收缩轮廓 (红色)", 0.8)
        self.showGlandCheck,  self.glandOpacitySlider  = create_visibility_row("乳腺腺体实质 (蓝色)", 0.5)
        self.showLesionCheck, self.lesionOpacitySlider = create_visibility_row("病灶靶点区域 (洋红)", 0.7)
        self.showHeatmapCheck, self.heatmapOpacitySlider = create_visibility_row("距离热力图 (红近蓝远)", 1.0)

        displayLayout.addRow(self.showBreastCheck, self.breastOpacitySlider)
        displayLayout.addRow(self.showErodedCheck, self.erodedOpacitySlider)
        displayLayout.addRow(self.showGlandCheck,  self.glandOpacitySlider)
        displayLayout.addRow(self.showLesionCheck, self.lesionOpacitySlider)
        displayLayout.addRow(self.showHeatmapCheck, self.heatmapOpacitySlider)

        # 绑定可见性控制事件
        self.showBreastCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Breast", state))
        self.showErodedCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Eroded", state))
        self.showGlandCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Gland", state))
        self.showLesionCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Lesion", state))
        self.showHeatmapCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Heatmap", state))

        # 绑定透明度控制事件
        self.breastOpacitySlider.connect('valueChanged(double)', lambda val: self.onChangeOpacity("Breast", val))
        self.erodedOpacitySlider.connect('valueChanged(double)', lambda val: self.onChangeOpacity("Eroded", val))
        self.glandOpacitySlider.connect('valueChanged(double)',  lambda val: self.onChangeOpacity("Gland", val))
        self.lesionOpacitySlider.connect('valueChanged(double)', lambda val: self.onChangeOpacity("Lesion", val))
        self.heatmapOpacitySlider.connect('valueChanged(double)', lambda val: self.onChangeOpacity("Heatmap", val))

        # 页面底部预留弹性拉伸空间
        self.layout.addStretch(1)

        # 初始检测场景是否有已提取的蓝色腺体
        existing_gland = slicer.mrmlScene.GetFirstNodeByName("Glands_KMeans_Extracted")
        if existing_gland:
            self.distGlandSelector.setCurrentNode(existing_gland)

    # =========================================================================
    # === 页面各组件对应的槽函数/事件响应 (Callbacks) ===
    # =========================================================================

    def onApplyBreast(self):
        inputVolume = self.inputSelector.currentNode()
        modelPath = self.breastModelSelector.currentPath
        if not inputVolume or not modelPath:
            slicer.util.errorDisplay("请确保已选择输入影像和乳房模型路径！")
            return
            
        logic = SimpleBreastDividerLogic()
        logic.run(inputVolume, modelPath, name_prefix="Breast", color=None, constrainMaskNode=None)
        self.showBreastCheck.checked = True

    def onApplyLesion(self):
        inputVolume = self.inputSelector.currentNode()
        modelPath = self.lesionModelSelector.currentPath
        constraintMask = self.lesionConstraintSelector.currentNode() 
        
        if not inputVolume or not modelPath:
            slicer.util.errorDisplay("请确保已选择输入影像和病灶模型路径！")
            return
            
        # 智能防呆设计：自动反向锁定乳房区域掩膜
        if not constraintMask:
            expected_mask_name = f"Breast_{inputVolume.GetName()}"
            auto_mask = slicer.mrmlScene.GetFirstNodeByName(expected_mask_name)
            
            if auto_mask:
                constraintMask = auto_mask
                self.lesionConstraintSelector.setCurrentNode(auto_mask)
                slicer.util.showStatusMessage(f"已自动套用空间约束: {expected_mask_name}")
                print(f"[INFO] 用户未选约束，系统已自动锁定并使用: {expected_mask_name}")
            else:
                slicer.util.warningDisplay("未检测到乳房掩膜！本次病灶分割将在无空间约束下进行，可能会产生假阳性。建议先执行【乳房区域分割】。")
            
        logic = SimpleBreastDividerLogic()
        logic.run(inputVolume, modelPath, name_prefix="Lesion", 
                  color=(1.0, 0.0, 1.0), constrainMaskNode=constraintMask)
        self.showLesionCheck.checked = True

    def onCalcVolume(self):
        maskNode = self.maskSelector.currentNode()
        erosion_mm = self.erosionSpinBox.value

        if not maskNode:
            slicer.util.errorDisplay("请先选择一个分割结果 (Mask)！")
            return

        slicer.util.showStatusMessage(f"正在以 {erosion_mm}mm 进行收缩计算...")
        slicer.app.processEvents()

        try:
            logic = SimpleBreastDividerLogic()
            left_ml, right_ml = logic.calculate_eroded_volume(maskNode, erosion_mm)
            
            self.leftResultTextBox.text = f"左侧: {left_ml:.3f} ml"
            self.rightResultTextBox.text = f"右侧: {right_ml:.3f} ml"
            
            slicer.util.showStatusMessage("体积计算完成！")
            
            if left_ml == 0 and right_ml == 0:
                slicer.util.warningDisplay(f"收缩 {erosion_mm}mm 后内容全空！请尝试减小收缩距离。")
            
            # 精准抓取生成的轮廓节点并自动联动下游下拉框
            eroded_node_name = f"Eroded_{erosion_mm}mm_Outline"
            eroded_node = slicer.mrmlScene.GetFirstNodeByName(eroded_node_name)
            if eroded_node:
                self.glandMaskSelector.setCurrentNode(eroded_node)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            slicer.util.errorDisplay(f"体积计算出错: {str(e)}")

    def onCalcGland(self):
        imageNode = self.glandImageSelector.currentNode()
        maskNode = self.glandMaskSelector.currentNode()

        if not imageNode or not maskNode:
            slicer.util.errorDisplay("请确保已选择原始影像和 Mask 节点！")
            return

        slicer.util.showStatusMessage("正在执行 K-Means 与形态学处理，请耐心等待...")
        slicer.app.processEvents()

        try:
            logic = SimpleBreastDividerLogic()
            left_ml, right_ml = logic.extract_gland_volume(imageNode, maskNode)
            
            self.leftGlandTextBox.text = f"左侧腺体: {left_ml:.3f} ml"
            self.rightGlandTextBox.text = f"右侧腺体: {right_ml:.3f} ml"
            
            slicer.util.showStatusMessage("腺体提取完成！")
            
            gland_node = slicer.mrmlScene.GetFirstNodeByName("Glands_KMeans_Extracted")
            if gland_node:
                self.distGlandSelector.setCurrentNode(gland_node)

        except Exception as e:
            import traceback
            traceback.print_exc()
            slicer.util.errorDisplay(f"腺体计算出错: {str(e)}")

    def onCalcHeatmap(self):
        breastNode = self.distBreastSelector.currentNode()
        glandNode = self.distGlandSelector.currentNode()
        
        if not breastNode or not glandNode:
            slicer.util.errorDisplay("请确保已选择乳房轮廓和腺体掩膜！")
            return
            
        slicer.util.showStatusMessage("正在提取 3D 网格并计算空间最短距离，请稍候...")
        slicer.app.processEvents()
        
        try:
            logic = SimpleBreastDividerLogic()
            logic.generate_distance_heatmap(breastNode, glandNode)
            
            # 可视化联动优化：自动隐藏纯色实质网格，突出彩色边界热力图
            self.showGlandCheck.setChecked(False)
            self.showHeatmapCheck.setChecked(True)
            self.heatmapOpacitySlider.value = 1.0
            
            slicer.util.showStatusMessage("热力图生成成功！红色代表极近区域。", 3000)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            slicer.util.errorDisplay(f"热力图计算出错: {str(e)}")
            slicer.util.showStatusMessage("热力图计算失败或中断！", 3000)

    def onToggleVisibility(self, prefix, state):
        isVisible = (state == 2)
        target_node_name = ""
        
        inputVolume = self.inputSelector.currentNode()
        if inputVolume and prefix not in ["Eroded", "Gland", "Heatmap"]:
            target_node_name = f"{prefix}_{inputVolume.GetName()}"
        elif prefix == "Eroded":
            erosion_mm = self.erosionSpinBox.value 
            target_node_name = f"Eroded_{erosion_mm}mm_Outline"
        elif prefix == "Gland":
            target_node_name = "Glands_KMeans_Extracted"
        elif prefix == "Heatmap":
            target_node_name = "Gland_Distance_Heatmap_Model"

        node = slicer.mrmlScene.GetFirstNodeByName(target_node_name)
        if node:
            displayNode = node.GetDisplayNode()
            if displayNode:
                if node.IsA("vtkMRMLSegmentationNode"):
                    displayNode.SetVisibility3D(isVisible)
                elif node.IsA("vtkMRMLModelNode"):
                    displayNode.SetVisibility(isVisible)
                    
                    # 热力图特殊处理：同步更新颜色数轴的显隐
                    if prefix == "Heatmap":
                        legendNode = slicer.mrmlScene.GetFirstNodeByName("Gland_Distance_Heatmap_Legend")
                        if legendNode:
                            legendNode.SetVisibility(isVisible)
        
    def onChangeOpacity(self, prefix, opacity_value):
        target_node_name = ""
        inputVolume = self.inputSelector.currentNode()
        if inputVolume and prefix not in ["Eroded", "Gland", "Heatmap"]:
            target_node_name = f"{prefix}_{inputVolume.GetName()}"
        elif prefix == "Eroded":
            erosion_mm = self.erosionSpinBox.value 
            target_node_name = f"Eroded_{erosion_mm}mm_Outline"
        elif prefix == "Gland":
            target_node_name = "Glands_KMeans_Extracted"
        elif prefix == "Heatmap":
            target_node_name = "Gland_Distance_Heatmap_Model"

        node = slicer.mrmlScene.GetFirstNodeByName(target_node_name)
        if node:
            displayNode = node.GetDisplayNode()
            if displayNode:
                if node.IsA("vtkMRMLSegmentationNode"):
                    segmentation = node.GetSegmentation()
                    for i in range(segmentation.GetNumberOfSegments()):
                        s_id = segmentation.GetNthSegmentID(i)
                        displayNode.SetSegmentOpacity3D(s_id, opacity_value)
                elif node.IsA("vtkMRMLModelNode"):
                    displayNode.SetOpacity(opacity_value)