# Social Reinforcement Learning in Networked Best-Shot Games


Simulation data and reproduction code for *Social Reinforcement Learning Drives Near-optimal Investment Patterns in Networked Best-Shot Games*.

- Lattice experiments:100×100 periodic grid,10000 agents, random asynchronous updates with replacement,10000 selections per MCS.
- Figure8:100-agent WS, BA and ER networks, with1650 completed algorithm runs.
- Original data and figures:CC BY4.0. Original code:MIT. Third-party UCT attribution and license are retained.

Files are organized by manuscript figure. See the sections below for commands, data descriptions and the scope of available outputs. Full8000-MCS reruns of the aligned lattice code are pending; existing tables remain stored study outputs.

本仓库对应论文 Social Reinforcement Learning Drives Near-optimal Investment Patterns in Networked Best-Shot Games。所有文件说明和运行方法集中在本README中，各目录按论文图号排列。

## 环境与通用参数

格点实验（图1、2、3、4、6、7）采用100×100周期网络，共10000个节点，每个节点有4个最近邻。每个MCS均匀、有放回地随机抽取10000次节点；选中节点的动作、邻居投资计数和Q值逐次立即更新。

动作0为不投资，1为投资。投资者阶段收益为−c；受邻居保护的不投资者为0；未被保护的不投资者为−1。投资邻居数0、1、2、3、4对应状态编号2、2、1、0、0。Q表全零初始化，并列最优动作均匀随机选择。alpha=0.1，gamma=0.95，epsilon=0.005；Q学习保留奖励与折扣未来价值项。完整演化默认8000 MCS，图4提取指定中间时刻快照。具体成本、权重和时刻见各main.py及参数文件。

安装Python3.10或以上。格点实验依赖：

```powershell
python -m pip install numpy pandas matplotlib networkx tqdm
```

以下命令均在解压后的数据包根目录执行。每张实验图只有一个main.py入口。

## Figure1

| 文件 | 内容 |
| --- | --- |
| main.py | 社会Q学习、个体Q学习、BR和Fermi的统一运行入口 |
| figure1_plot_data.csv | 从原Origin工程导出的绘图表，保留原列顺序 |
| BR_summary.csv | 修正为随机异步更新的BR均值、标准差、标准误 |
| BR_individual_runs.csv | 12个成本、每个成本5次实验，共60次BR实验的结果 |
| BR_final_states.npz | 60次BR实验最终配置，合并在一个文件中；每份为100×100的0/1数组 |
| parameters.json | 参数、种子及原Origin列和图的对应信息 |
| figure1.pdf | 参考图 |

运行：`python figure1/main.py`

新结果写入figure1/new_results/。BR_reward为平均社会收益，BR_investment为投资比例；std为样本标准差，se为标准误。使用numpy.load读取BR_final_states.npz，键名如c00_run01表示成本列表索引和重复次数。原Origin表中未命名的列保留原名称，图线对应信息见parameters.json。

## Figure2

- main.py：实验和绘图入口。
- plot_data.csv：Origin绘图宽表。
- parameter_results.csv：各参数组合的汇总结果，不是逐次独立实验记录。
- origin_table.zip：原Origin工作表及列说明。
- figure2.pdf：参考图。

运行：`python figure2/main.py`

## Figure3

- main.py：实验、数值导出和绘图入口。
- plot_data.csv：已保存的收益演化曲线数值。
- display_data.npz：从参考PDF提取的曲线路径及快照图像。
- parameters.json：显示数据的来源和对应参数。
- figure3.pdf：参考图。

运行：`python figure3/main.py`

## Figure4

- main.py：实验和快照绘图入口，运行后导出新的节点配置。
- figure4.pdf：参考图。

运行：`python figure4/main.py`

## Figure5

figure5.pdf为机制示意图，没有独立的数值实验数据。

## Figure6

- main.py：实验、构型比例统计、数值导出和绘图入口。
- plot_data.csv：已保存的构型比例演化曲线。
- display_data.npz：从参考PDF提取的曲线路径及快照图像。
- parameters.json：显示数据的来源和对应参数。
- figure6.pdf：参考图。

运行：`python figure6/main.py`

## Figure7

- main.py：给定初始投资密度及汉明扰动下的BR和Q学习实验，运行后导出数值及节点配置。
- figure7.pdf：参考图。

运行：`python figure7/main.py`

## Figure8

| 文件 | 内容 |
| --- | --- |
| main.py | 实验、数值导出和绘图的统一入口 |
| plot_data.csv | 三种网络的Origin绘图表合并数据，network列区分WS、BA、ER |
| runs.csv | 1650次实验的逐次结果 |
| final_states.jsonl.gz | 1650份最终配置合并压缩，每行为一份JSON，record_id标识原实验 |
| networks.zip | 三份网络拓扑，GraphML格式 |
| parameters.json | 实际运行参数、随机种子规则和上游版本 |
| uct_source.zip | 必要UCT源码、上游许可证及兼容性补丁，Python直接导入，无需解压 |
| requirements.txt | 运行依赖 |
| figure8.png | 结果图 |

本图为单独的100节点复杂网络实验。WS的k=4、p=0.1，BA的m=2，ER的p=0.2。随机异步有放回更新。完整实验为3种网络×11个成本×5种算法×10次重复，演化8000 MCS，最后100 MCS取平均。

```powershell
python -m pip install -r figure8/requirements.txt
python figure8/main.py --network-seeds 2 --runs 10 --steps 8000 --output figure8/new_results
```

仅根据包内现有数值绘图：

```powershell
python figure8/main.py --plot-only
```

pay为平均社会收益，inv为投资比例，std为样本标准差。Fermi在论文中记为FR。final_states.jsonl.gz可使用Python的gzip.open以文本方式逐行读取，再用json.loads解析。

## Figure9

- main.py：直接读取图8已保存的网络和SPQ最终配置，绘制三个拓扑的快照，无需重复模拟。
- figure9.pdf：论文原参考快照。

运行：`python figure9/main.py --cost 0.5 --run 0`

默认使用网络种子2、成本0.5、重复编号0，按预先固定的实验编号取样，不挑选收益最高的重复。可用--cost和--run指定其他已存实验。新图及其配置保存到figure9/new_results/，不会覆盖figure9.pdf。选择记录同时列出终态指标和最后100 MCS均值，两者按实际含义分别保存。

本入口使用新图8实验的最终配置；尚未确认与论文旧图9逐节点相同。相关终态绘图函数在旧项目的复杂网络实验脚本中找到，但其原始快照的完整对应关系未确定。

## 数据版本与文件核验

本次整理合并文件和代码入口，没有改变已存实验数值。新版格点代码已完成实际100×100规模的短程核验；完整8000 MCS格点实验尚未重跑。已完成的修正BR实验和图8实验数值包含在相应目录。

图3、图6的display_data.npz保存图形路径和RGB快照，不是原始完整演化轨迹或0/1节点状态数组。各图可用材料如上所列。

各图目录集中保存代码、已存数据和参考图；本README统一说明其用途和运行方法。实验参数见本文及对应脚本。

## 许可证与引用

UCT代码来源：Darvariu、Hailes和Musolesi，https://github.com/VictorDarvariu/solving-graph-pgg 。上游MIT许可证、原始署名和兼容性补丁保存在figure8/uct_source.zip中。

本数据包中作者自有的实验数据和原创图表采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可证；作者自有代码及相关软件文档采用 [MIT](https://opensource.org/license/mit) 许可证。许可证条款、官方链接及适用范围见根目录 LICENSE.txt。第三方代码保留其原有版权声明、署名和许可证。
