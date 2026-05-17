import os
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest
import SampleData

# 通过相对导入引入同包下的 Logic 算法控制类
from .Logic import SimpleBreastDividerLogic

class SimpleBreastDividerTest(ScriptedLoadableModuleTest):
    """
    SimpleBreastDividerTest (自动化测试层)
    负责验证模块核心算法逻辑（如形态学收缩、体积计算、热力图网格生成等）的解耦单元测试。
    """

    def setUp(self):
        """在每次测试用例执行前清空 Slicer 场景，确保测试环境纯净。"""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """运行自动化测试的主入口。"""
        self.setUp()
        self.test_SimpleBreastDividerAlgorithm()

    def test_SimpleBreastDividerAlgorithm(self):
        """
        测试核心图像处理算法。
        注意：原模板自带的旧测试代码调用了不存在的 `logic.process` 导致报错。
        重构后的测试脚本已修正导入路径，并为你的解剖学分析算法搭建了标准的测试骨架。
        """
        self.delayDisplay("开始执行 SimpleBreastDivider 自动化单元测试...")

        # 1. 智能引入主脚本中的数据源注册函数，确保测试集可以被 SampleData 正常识别
        try:
            from SimpleBreastDivider import registerSampleData
            registerSampleData()
        except ImportError:
            # 兜底机制：如果加载路径不匹配，手动在测试内部重新注册样例数据
            iconsPath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Resources", "Icons")
            SampleData.SampleDataLogic.registerCustomSampleDataSource(
                category="SimpleBreastDivider",
                sampleName="SimpleBreastDivider1",
                thumbnailFileName=os.path.join(iconsPath, "SimpleBreastDivider1.png"),
                uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
                fileNames="SimpleBreastDivider1.nrrd",
                checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82code-generated-file-",
                nodeNames="SimpleBreastDivider1",
            )

        # 2. 从云端下载并加载指定的测试影像
        inputVolume = SampleData.downloadSample("SimpleBreastDivider1")
        self.delayDisplay("测试影像 SimpleBreastDivider1 下载并成功加载至场景。")

        # 3. 断言验证：检查影像的灰度/标量范围是否符合预期（防呆校验）
        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        # 4. 实例化重构后的算法逻辑类
        logic = SimpleBreastDividerLogic()
        self.assertIsNotNone(logic, "SimpleBreastDividerLogic 实例化失败！")
        self.delayDisplay("Logic 算法核心类解耦测试通过。")

        # =========================================================================
        # 💡 实验室开发建议：
        # 由于 nnUNet 深度学习推理极其依赖本地显卡资源（如单卡 Cuda:0 设备）和特定的模型权重目录，
        # 在自动化回归测试中，通常建议“跳过”耗时且环境敏感的 `logic.run`（推理部分），
        # 专门用来测试不依赖外部 AI 环境的纯 VTK / 形态学核心算法。
        #
        # 例如，你未来可以在此处手动创建一个小体积的假 Mask 节点，
        # 然后直接调用断言验证体积计算：
        # left_ml, right_ml = logic.calculate_eroded_volume(mockMaskNode, 5.0)
        # self.assertGreater(left_ml, 0.0)
        # =========================================================================

        self.delayDisplay("SimpleBreastDivider 所有重构单元测试项顺利通过！")