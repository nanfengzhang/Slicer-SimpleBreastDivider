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
        # --- 相对路径自动推导 ---
        # =========================================================================
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))

        default_breast_model = os.path.join(root_dir, "Dataset001_Breast", "nnUNetTrainer__nnUNetPlans__3d_fullres").replace('\\', '/')
        default_lesion_model = os.path.join(root_dir, "Dataset102_BreastLesion", "nnUNetTrainer__nnUNetPlans__3d_fullres").replace('\\', '/')

        # =========================================================================
        # === 核心容器：简约界面 (默认展开) vs 详细界面 (默认折叠) ===
        # =========================================================================
        self.simpleCollapsibleButton = ctk.ctkCollapsibleButton()
        self.simpleCollapsibleButton.text = "简约界面"
        self.simpleCollapsibleButton.collapsed = False  
        self.layout.addWidget(self.simpleCollapsibleButton)
        simpleLayout = qt.QVBoxLayout(self.simpleCollapsibleButton)

        self.detailedCollapsibleButton = ctk.ctkCollapsibleButton()
        self.detailedCollapsibleButton.text = "详细界面 (进阶设置)"
        self.detailedCollapsibleButton.collapsed = True  
        self.layout.addWidget(self.detailedCollapsibleButton)
        detailedLayout = qt.QVBoxLayout(self.detailedCollapsibleButton)

        # =========================================================================
        # === 模块 1：基础设置与 AI 分割执行区 ===
        # =========================================================================
        # 【分配到简约区】
        s_parametersButton = ctk.ctkCollapsibleButton()
        s_parametersButton.text = "1. 分割模型配置与执行"
        simpleLayout.addWidget(s_parametersButton)
        s_parametersLayout = qt.QFormLayout(s_parametersButton)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False; self.inputSelector.removeEnabled = False; self.inputSelector.noneEnabled = False; self.inputSelector.showHidden = False; self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.toolTip = "选择要预测的输入影像"
        s_parametersLayout.addRow("输入影像: ", self.inputSelector)

        self.applyBreastButton = qt.QPushButton("乳房区域分割")
        self.applyBreastButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #FFC107; color: white;")
        s_parametersLayout.addRow(self.applyBreastButton)

        # 【分配到详细区】
        d_parametersButton = ctk.ctkCollapsibleButton()
        d_parametersButton.text = "高级：模型路径与病灶分割"
        detailedLayout.addWidget(d_parametersButton)
        d_parametersLayout = qt.QFormLayout(d_parametersButton)

        self.breastModelSelector = ctk.ctkPathLineEdit()
        self.breastModelSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.breastModelSelector.currentPath = default_breast_model
        d_parametersLayout.addRow("乳房区域分割模型路径: ", self.breastModelSelector)

        self.lesionModelSelector = ctk.ctkPathLineEdit()
        self.lesionModelSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.lesionModelSelector.currentPath = default_lesion_model
        d_parametersLayout.addRow("病灶区域分割模型路径: ", self.lesionModelSelector)

        self.lesionConstraintSelector = slicer.qMRMLNodeComboBox()
        self.lesionConstraintSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode"]
        self.lesionConstraintSelector.addEnabled = False; self.lesionConstraintSelector.renameEnabled = True
        self.lesionConstraintSelector.setMRMLScene(slicer.mrmlScene)
        self.lesionConstraintSelector.toolTip = "选择乳腺掩膜，将病灶分割限制在其范围内"
        d_parametersLayout.addRow("空间约束掩膜 (乳腺): ", self.lesionConstraintSelector)
        
        self.applyLesionButton = qt.QPushButton("病灶区域分割")
        self.applyLesionButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #F44336; color: white;")
        d_parametersLayout.addRow(self.applyLesionButton)


        # =========================================================================
        # === 模块 2：乳房体积收缩分析 ===
        # =========================================================================
        # 【分配到简约区】
        s_volumeButton = ctk.ctkCollapsibleButton()
        s_volumeButton.text = "2. 乳房体积收缩分析"
        simpleLayout.addWidget(s_volumeButton)
        s_volumeLayout = qt.QFormLayout(s_volumeButton)

        self.erosionSpinBox = qt.QDoubleSpinBox()
        self.erosionSpinBox.setValue(5.0); self.erosionSpinBox.suffix = " mm"
        s_volumeLayout.addRow("边缘收缩距离: ", self.erosionSpinBox)

        self.calcVolumeButton = qt.QPushButton("乳房收缩体积分割及计算")
        self.calcVolumeButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #4CAF50; color: white;")
        s_volumeLayout.addRow(self.calcVolumeButton)

        self.leftResultTextBox = qt.QLineEdit(); self.leftResultTextBox.setReadOnly(True); self.leftResultTextBox.text = " 0.00 ml "
        s_volumeLayout.addRow("左侧乳房收缩后体积: ", self.leftResultTextBox)

        self.rightResultTextBox = qt.QLineEdit(); self.rightResultTextBox.setReadOnly(True); self.rightResultTextBox.text = " 0.00 ml "
        s_volumeLayout.addRow("右侧乳房收缩后体积: ", self.rightResultTextBox)

        # 【分配到详细区】
        d_volumeButton = ctk.ctkCollapsibleButton()
        d_volumeButton.text = "高级：修改体积计算输入源"
        detailedLayout.addWidget(d_volumeButton)
        d_volumeLayout = qt.QFormLayout(d_volumeButton)

        self.maskSelector = slicer.qMRMLNodeComboBox()
        self.maskSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode"]
        self.maskSelector.selectNodeUponCreation = True
        self.maskSelector.addEnabled = False; self.maskSelector.removeEnabled = False; self.maskSelector.noneEnabled = False; self.maskSelector.showHidden = False; self.maskSelector.showChildNodeTypes = False
        self.maskSelector.setMRMLScene(slicer.mrmlScene)
        d_volumeLayout.addRow("选择分割结果: ", self.maskSelector)


        # =========================================================================
        # === 模块 3：乳腺腺体分割与体积计算 ===
        # =========================================================================
        # 【分配到简约区】
        s_glandButton = ctk.ctkCollapsibleButton()
        s_glandButton.text = "3. 乳腺腺体分割与体积计算"
        simpleLayout.addWidget(s_glandButton)
        s_glandLayout = qt.QFormLayout(s_glandButton)

        self.calcGlandButton = qt.QPushButton("乳腺体积分割及计算")
        self.calcGlandButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #2196F3; color: white;")
        s_glandLayout.addRow(self.calcGlandButton)

        self.leftGlandTextBox = qt.QLineEdit(); self.leftGlandTextBox.setReadOnly(True); self.leftGlandTextBox.text = " 0.00 ml "
        s_glandLayout.addRow("左侧腺体体积: ", self.leftGlandTextBox)

        self.rightGlandTextBox = qt.QLineEdit(); self.rightGlandTextBox.setReadOnly(True); self.rightGlandTextBox.text = " 0.00 ml "
        s_glandLayout.addRow("右侧腺体体积: ", self.rightGlandTextBox)

        # 【分配到详细区】
        d_glandButton = ctk.ctkCollapsibleButton()
        d_glandButton.text = "高级：修改腺体计算输入源"
        detailedLayout.addWidget(d_glandButton)
        d_glandLayout = qt.QFormLayout(d_glandButton)

        self.glandImageSelector = slicer.qMRMLNodeComboBox()
        self.glandImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.glandImageSelector.selectNodeUponCreation = True
        self.glandImageSelector.addEnabled = False; self.glandImageSelector.removeEnabled = False; self.glandImageSelector.noneEnabled = False; self.glandImageSelector.showHidden = False; self.glandImageSelector.showChildNodeTypes = False
        self.glandImageSelector.setMRMLScene(slicer.mrmlScene)
        d_glandLayout.addRow("选择原始影像: ", self.glandImageSelector)

        self.glandMaskSelector = slicer.qMRMLNodeComboBox()
        self.glandMaskSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode"]
        self.glandMaskSelector.selectNodeUponCreation = True
        self.glandMaskSelector.addEnabled = False; self.glandMaskSelector.removeEnabled = False; self.glandMaskSelector.noneEnabled = False; self.glandMaskSelector.showHidden = False; self.glandMaskSelector.showChildNodeTypes = False
        self.glandMaskSelector.setMRMLScene(slicer.mrmlScene)
        d_glandLayout.addRow("选择乳腺范围(Mask): ", self.glandMaskSelector)


        # =========================================================================
        # === 模块 4：乳房表面安全边界分析 (3D 热力图) ===
        # =========================================================================
        # 【分配到简约区】
        s_distanceButton = ctk.ctkCollapsibleButton()
        s_distanceButton.text = "4. 乳房表面安全边界分析 (3D 热力图)"
        simpleLayout.addWidget(s_distanceButton)
        s_distanceLayout = qt.QFormLayout(s_distanceButton)

        self.calcDistanceButton = qt.QPushButton("生成 3D 距离热力图")
        self.calcDistanceButton.setStyleSheet("font-weight: bold; padding: 5px; background-color: #9C27B0; color: white;")
        s_distanceLayout.addRow(self.calcDistanceButton)

        # 【分配到详细区】
        d_distanceButton = ctk.ctkCollapsibleButton()
        d_distanceButton.text = "高级：修改 3D 热力图图层"
        detailedLayout.addWidget(d_distanceButton)
        d_distanceLayout = qt.QFormLayout(d_distanceButton)

        self.distTargetSelector = slicer.qMRMLNodeComboBox()
        self.distTargetSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLModelNode"]
        self.distTargetSelector.selectNodeUponCreation = True
        self.distTargetSelector.addEnabled = False; self.distTargetSelector.removeEnabled = False; self.distTargetSelector.noneEnabled = False; self.distTargetSelector.showHidden = False
        self.distTargetSelector.setMRMLScene(slicer.mrmlScene)
        self.distTargetSelector.setToolTip("请选择目标层：计算结果将作为热力图绘制在此结构表面（默认选择外部乳房）。")
        d_distanceLayout.addRow("目标层 (渲染表面): ", self.distTargetSelector)

        self.distRefSelector = slicer.qMRMLNodeComboBox()
        self.distRefSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLModelNode"]
        self.distRefSelector.selectNodeUponCreation = True
        self.distRefSelector.addEnabled = False; self.distRefSelector.removeEnabled = False; self.distRefSelector.noneEnabled = False; self.distRefSelector.showHidden = False
        self.distRefSelector.setMRMLScene(slicer.mrmlScene)
        self.distRefSelector.setToolTip("请选择参照层：计算目标层到此结构的距离（默认选择内部腺体/病灶）。")
        d_distanceLayout.addRow("参照层 (测距基准): ", self.distRefSelector)


        # =========================================================================
        # === 模块 5：可视化与 3D 视图层级管理面板 ===
        # =========================================================================
        # 视图管理完全保留在简约界面中
        self.displayCollapsibleButton = ctk.ctkCollapsibleButton()
        self.displayCollapsibleButton.text = "5. 3D 视图层级管理"
        simpleLayout.addWidget(self.displayCollapsibleButton)
        displayLayout = qt.QFormLayout(self.displayCollapsibleButton)

        def create_visibility_row(label_text, default_opacity):
            checkbox = qt.QCheckBox(label_text); checkbox.checked = True
            slider = ctk.ctkSliderWidget()
            slider.singleStep = 0.1; slider.minimum = 0.0; slider.maximum = 1.0; slider.value = default_opacity; slider.decimals = 2
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

        # =========================================================================
        # === 事件绑定 (完全保留原有映射，绝不触碰) ===
        # =========================================================================
        self.applyBreastButton.connect('clicked(bool)', self.onApplyBreast)
        self.applyLesionButton.connect('clicked(bool)', self.onApplyLesion)
        self.calcVolumeButton.connect('clicked(bool)', self.onCalcVolume)
        self.calcGlandButton.connect('clicked(bool)', self.onCalcGland)
        self.calcDistanceButton.connect('clicked(bool)', self.onCalcHeatmap)

        self.showBreastCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Breast", state))
        self.showErodedCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Eroded", state))
        self.showGlandCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Gland", state))
        self.showLesionCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Lesion", state))
        self.showHeatmapCheck.stateChanged.connect(lambda state: self.onToggleVisibility("Heatmap", state))

        self.breastOpacitySlider.connect('valueChanged(double)', lambda val: self.onChangeOpacity("Breast", val))
        self.erodedOpacitySlider.connect('valueChanged(double)', lambda val: self.onChangeOpacity("Eroded", val))
        self.glandOpacitySlider.connect('valueChanged(double)',  lambda val: self.onChangeOpacity("Gland", val))
        self.lesionOpacitySlider.connect('valueChanged(double)', lambda val: self.onChangeOpacity("Lesion", val))
        self.heatmapOpacitySlider.connect('valueChanged(double)', lambda val: self.onChangeOpacity("Heatmap", val))

        self.layout.addStretch(1)

        existing_gland = slicer.mrmlScene.GetFirstNodeByName("Glands_KMeans_Extracted")
        if existing_gland:
            self.distRefSelector.setCurrentNode(existing_gland)

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
            
            self.leftResultTextBox.text = f" {left_ml:.3f} ml"
            self.rightResultTextBox.text = f" {right_ml:.3f} ml"
            
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

        slicer.util.showStatusMessage("正在执行腺体分割及计算处理，请耐心等待...")
        slicer.app.processEvents()

        try:
            logic = SimpleBreastDividerLogic()
            left_ml, right_ml = logic.extract_gland_volume(imageNode, maskNode)
            
            self.leftGlandTextBox.text = f" {left_ml:.3f} ml"
            self.rightGlandTextBox.text = f" {right_ml:.3f} ml"
            
            slicer.util.showStatusMessage("腺体提取完成！")
            
            gland_node = slicer.mrmlScene.GetFirstNodeByName("Glands_KMeans_Extracted")
            if gland_node:
                self.distRefSelector.setCurrentNode(gland_node)

        except Exception as e:
            import traceback
            traceback.print_exc()
            slicer.util.errorDisplay(f"腺体计算出错: {str(e)}")

    # def onCalcHeatmap(self):
    #     breastNode = self.distBreastSelector.currentNode()
    #     glandNode = self.distGlandSelector.currentNode()
        
    #     if not breastNode or not glandNode:
    #         slicer.util.errorDisplay("请确保已选择乳房轮廓和腺体掩膜！")
    #         return
            
    #     slicer.util.showStatusMessage("正在提取 3D 网格并计算空间最短距离，请稍候...")
    #     slicer.app.processEvents()
        
    #     try:
    #         logic = SimpleBreastDividerLogic()
    #         logic.generate_distance_heatmap(breastNode, glandNode)
            
    #         # 可视化联动优化：自动隐藏纯色实质网格，突出彩色边界热力图
    #         self.showGlandCheck.setChecked(False)
    #         self.showHeatmapCheck.setChecked(True)
    #         self.heatmapOpacitySlider.value = 1.0
            
    #         slicer.util.showStatusMessage("热力图生成成功！红色代表极近区域。", 3000)
            
    #     except Exception as e:
    #         import traceback
    #         traceback.print_exc()
    #         slicer.util.errorDisplay(f"热力图计算出错: {str(e)}")
    #         slicer.util.showStatusMessage("热力图计算失败或中断！", 3000)

    def onCalcHeatmap(self):
        # 1. 提取新重构的抽象层节点
        targetNode = self.distTargetSelector.currentNode()
        refNode = self.distRefSelector.currentNode()
        
        if not targetNode or not refNode:
            slicer.util.errorDisplay("请同时选择目标层和参照层！")
            return
            
        slicer.util.showStatusMessage("正在提取 3D 网格并计算空间最短绝对距离，请稍候...")
        slicer.app.processEvents() # 保持你原本优秀的防假死设计
        
        try:
            from SimpleBreastDividerLib.Logic import SimpleBreastDividerLogic
            logic = SimpleBreastDividerLogic()
            
            # 2. 调用修改后的逻辑层算法方法
            logic.processDistanceHeatmap(targetNode, refNode)
            
            self.lastHeatmapName = f"{targetNode.GetName()}_to_{refNode.GetName()}_Heatmap"
            
            # 3. 可视化联动优化 (完美保留你的原版交互)
            # 自动隐藏纯色实质网格，突出彩色边界热力图
            self.showGlandCheck.setChecked(False)
            self.showHeatmapCheck.setChecked(True)
            self.heatmapOpacitySlider.value = 1.0
            
            slicer.util.showStatusMessage("热力图生成成功！红色代表极近区域。", 3000)

            # ⭐【新增这一行】：生成完后，主动触发一次刷新，让滑块和复选框立刻绑定新模型
           # 2 代表复选框的 Checked (勾选) 状态
            self.onToggleVisibility("Heatmap", 2) 
            
            # 1.0 代表不透明度全开
            self.onChangeOpacity("Heatmap", 1.0)
            
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
            # ⭐ 动态获取热力图名称。使用 getattr 防呆，避免用户没生成热力图前乱点报错
            target_node_name = getattr(self, "lastHeatmapName", "Gland_Distance_Heatmap_Model")

        node = slicer.mrmlScene.GetFirstNodeByName(target_node_name)
        if node:
            displayNode = node.GetDisplayNode()
            if displayNode:
                if node.IsA("vtkMRMLSegmentationNode"):
                    displayNode.SetVisibility3D(isVisible)
                elif node.IsA("vtkMRMLModelNode"):
                    displayNode.SetVisibility(isVisible)
                    
                    # ⭐ 热力图特殊处理：同步更新颜色数轴的显隐
                    if prefix == "Heatmap":
                        # 数轴的名字在 Logic 中被命名为: f"{model_name}_Legend"
                        legend_name = f"{target_node_name}_Legend"
                        legendNode = slicer.mrmlScene.GetFirstNodeByName(legend_name)
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
            # ⭐ 动态获取热力图名称
            target_node_name = getattr(self, "lastHeatmapName", "Gland_Distance_Heatmap_Model")

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