import pandas as pd
import re

# 读取 TSV 文件
df = pd.read_csv(
    r"C:\HiWi_Hanyang\ontology_matching\automatic-semantic-annotation\energy_domain_experiment\SORBET\Sargon2sbeo_-1.tsv",
    sep="\t", header=None)

# 只取前两列
df = df.iloc[:, :2]


# 清理函数：去掉 URL、命名空间前缀，只留最后的名称
def clean_value(x):
    if not isinstance(x, str):
        return x
    # 去掉 URL 前缀或命名空间前缀
    x = re.sub(r'.*[#/]', '', x)  # 去掉最后一个 '/' 或 '#' 前面的内容
    x = re.sub(r'.*:', '', x)  # 去掉最后一个 ':' 前面的内容
    return x.strip()


# 应用清理函数
df = df.applymap(clean_value)

# 保存为 Excel 文件
df.to_excel(
    r"C:\HiWi_Hanyang\ontology_matching\automatic-semantic-annotation\energy_domain_experiment\SORBET\Sargon2sbeo_-1.xlsx",
    index=False, header=True)

print("转换完成：output.xlsx")
