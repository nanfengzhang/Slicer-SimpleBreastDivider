import os
import logging
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _
from slicer.i18n import translate

# =========================================================================
# 1. 模块注册主类 (Module Registration Class)
# =========================================================================

class SimpleBreastDivider(ScriptedLoadableModule):
    """
    SimpleBreastDivider 模块的主入口类。
    负责在 3D Slicer 启动时向系统注册模块的元数据、分类、贡献者以及帮助文档。
    该类不处理任何 UI 渲染或算法逻辑。
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        
        # 优化模块在 UI 模块选择器中的人类可读名称
        self.parent.title = _("Simple Breast Divider")  
        
        # 将模块分拨到更专业的分类目录下
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Computational Medicine")]
        self.parent.dependencies = []  
        self.parent.contributors = ["Graduate Student (Computational Medicine & Medical Image Processing)"]
        
        # 完善模块在线文档与帮助提示
        self.parent.helpText = _("""
本模块提供了一套用于乳腺多模态分析的高级工具集，包括：<br>
1. <b>nnUNet 深度学习分割：</b>一键执行全自动乳房及病灶区域分割。<br>
2. <b>空间约束过滤：</b>自动将病灶分割结果限制在解剖学乳腺掩膜范围内，消除假阳性。<br>
3. <b>形态学边缘收缩：</b>精准计算解剖学上的乳腺体积上限值。<br>
4. <b>K-Means 腺体提取：</b>全自动计算真实腺体实质体积。<br>
5. <b>3D 安全边界距离热力图：</b>无缝计算内部腺体到最外层皮肤轮廓的空间 3D 最短距离。
""")
        
        self.parent.acknowledgementText = _("""
本模块基于 3D Slicer Scripted Loadable Module 模板进行架构重构。
核心逻辑采用了解耦设计，以便于后续深度学习模型的升级与环境维护。
""")

        # 监听 3D Slicer 应用程序启动完成信号，注册自带的测试数据集
        slicer.app.connect("startupCompleted()", registerSampleData)


# =========================================================================
# 2. 样本数据注册函数 (Sample Data Registration)
# =========================================================================

def registerSampleData():
    """
    在系统的 Sample Data 模块中注册定制化的样例数据，方便快速试用。
    """
    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons")
    
    # 自动创建缺少的资源目录，防止因找不到图标路径而报错
    if not os.path.exists(iconsPath):
        os.makedirs(iconsPath, exist_ok=True)

    # 注册样例数据集 1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="SimpleBreastDivider",
        sampleName="SimpleBreastDivider1",
        thumbnailFileName=os.path.join(iconsPath, "SimpleBreastDivider1.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SimpleBreastDivider1.nrrd",
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        nodeNames="SimpleBreastDivider1",
    )

    # 注册样例数据集 2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="SimpleBreastDivider",
        sampleName="SimpleBreastDivider2",
        thumbnailFileName=os.path.join(iconsPath, "SimpleBreastDivider2.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SimpleBreastDivider2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        nodeNames="SimpleBreastDivider2",
    )


# =========================================================================
# 3. 核心解耦重构：动态导入外部子包，并将其暴露给 Slicer 动态加载器
# =========================================================================
# 3D Slicer 会利用 Python 动态反射机制，在模块主脚本的全局命名空间中寻找以
# [ModuleName]Widget、[ModuleName]Logic 和 [ModuleName]Test 命名的类。
# 通过将其移至 `SimpleBreastDividerLib/` 独立文件并通过以下方式导入，
# 既满足了 Slicer 的加载标准，又实现了完美的物理代码隔离。

try:
    from SimpleBreastDividerLib.ParameterNode import SimpleBreastDividerParameterNode
except ImportError as e:
    logging.warning(f"[SimpleBreastDivider] 未检测到独立的 ParameterNode 脚本。提示: {e}")

try:
    from SimpleBreastDividerLib.Widget import SimpleBreastDividerWidget
except ImportError as e:
    logging.critical(f"[SimpleBreastDivider] 核心错误：无法从 SimpleBreastDividerLib.Widget 导入 UI 界面类！")
    logging.critical(f"请检查 SimpleBreastDividerLib/Widget.py 文件是否存在。详细错误: {e}")

try:
    from SimpleBreastDividerLib.Logic import SimpleBreastDividerLogic
except ImportError as e:
    logging.critical(f"[SimpleBreastDivider] 核心错误：无法从 SimpleBreastDividerLib.Logic 导入算法逻辑类！")
    logging.critical(f"请检查 SimpleBreastDividerLib/Logic.py 文件是否存在。详细错误: {e}")

try:
    from SimpleBreastDividerLib.Testing import SimpleBreastDividerTest
except ImportError:
    pass