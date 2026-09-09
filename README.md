# 碳化硅与硅外延层厚度的红外干涉测量

本仓库管理 2025 年全国大学生数学建模竞赛 B 题的研究代码、原始光谱、计算结果、图表和 LaTeX 论文。

论文、求解代码及复现材料已完成。当前只交付 LaTeX 源码，不生成 Word 或 PDF。所给题目实际标注年份为 2025。

- [论文主文件](B题论文/main.tex)
- [运行方法与结果说明](B题论文/README.md)
- [数值和源码核验记录](B题论文/results/verification.json)
- [LaTeX 编译检查记录](B题论文/results/latex_check.json)

主模型下，碳化硅共同厚度约 **7.38 μm**，硅共同厚度约 **3.41 μm**。完整项目包含四份原始数据、15张科学图、19张表、15条实际引用文献，以及分块验证、重抽样和多因素敏感性分析。厚度依赖文中明确说明的折射率与材料响应假设。

一键复现：

```bash
cd B题论文
python -m pip install -r requirements.txt
python run_all.py
```

使用用户提供的 LaTeX 模板，保留主体版式，替换为 TeX 发行版字体。详细验证范围见项目说明；仓库不包含 Word/PDF 成品。
