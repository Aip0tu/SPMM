# SPMM：面向分子的结构-性质多模态学习

这是 SPMM 的官方 GitHub 仓库。SPMM 是一个多模态分子预训练模型，用于协同理解分子结构与性质。
详细内容可参考以下论文：
*Bidirectional Generation of Structure and Properties Through a Single Molecular Foundation Model. ([Nature Communications 2024](https://www.nature.com/articles/s41467-024-46440-3))*

[![DOI](https://zenodo.org/badge/542878783.svg)](https://zenodo.org/doi/10.5281/zenodo.10567598)

***

![method1](https://github.com/jinhojsk515/SPMM/assets/59189526/1ff52950-aa12-481f-94ea-4d1e97ac7bf3)

分子结构以 SMILES 表示，我们使用 53 个简单化学性质构建分子的性质向量（PV）。

***<ins>模型检查点和数据体积较大，未直接包含在本仓库中，可从[这里](https://drive.google.com/drive/folders/1ARrSg9kXdXAL5VGgDBwizpSgcJwauPua?usp=sharing)下载。<ins>***

## 文件说明
* `data/`：保存论文实验所使用的数据。（需要你自行创建该目录，并放入上方链接下载的数据。）
* `Pretrain/`：保存预训练好的 SPMM 检查点。（需要你自行创建该目录，并放入上方链接下载的检查点。）
* `vocab_bpe_300.txt`：SMILES 分词器使用的 SMILES token 表。
* `property_name.txt`：53 个化学性质的名称。
* `normalize.pkl`：构建 PV 时使用的 53 个化学性质的均值与标准差。
* `calc_property.py`：用于计算 53 个化学性质，并根据给定 SMILES 构建 PV。**如果你要将 SPMM 预训练用于自定义 PV，请按需修改此文件。**
* `SPMM_models.py`：SPMM 模型及其预训练代码。
* `SPMM_pretrain.py`：用于运行 SPMM 预训练。
* `d_*.py`：下游任务脚本。

## 环境依赖
运行 `pip install -r requirements.txt` 安装所需依赖。

## 运行方法
参数既可以通过命令行传入，也可以直接在脚本中手动修改。

1. 预训练
    ```
    python SPMM_pretrain.py --data_path './data/pretrain.txt'
    ```

2. PV 到 SMILES 生成
   * batched：模型会读取 `input_file` 中分子的 PV，并使用 k-beam search 生成具有这些 PV 的分子。生成结果会写入 `generated_molecules.txt`。
       ```
       python d_pv2smiles_batched.py --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --input_file './data/pubchem_1k_unseen.txt' --k 2
       ```
   * single：模型接收一个查询 PV，并使用 k-beam search 生成 `n_generate` 个满足该 PV 的分子。生成结果会写入 `generated_molecules.txt`。你需要先在 `p2s_input.csv` 中构建输入 PV，可参考仓库提供的四个示例。
       ```
       python d_pv2smiles_single.py --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --n_generate 1000 --stochastic True --k 2
       ```

3. SMILES 到 PV 生成

    模型会读取 `input_file` 中的查询分子，并生成对应的 PV。

    ```
    python d_smiles2pv.py --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --input_file './data/pubchem_1k_unseen.txt'
    ```

4. MoleculeNet + DILI 预测任务

    `d_regression.py`、`d_classification.py` 和 `d_classification_multilabel.py` 分别对应回归、二分类和多标签分类任务。

    ```
    python d_regression.py --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --name 'bace'
    python d_classification.py --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --name 'bbbp'
    python d_classification_multilabel.py --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --name 'clintox'
    ```

5. 正向/逆向反应预测任务

    `d_rxn_prediction.py` 可在 USPTO-480k 和 USPTO-50k 数据集上执行正向反应预测与逆合成预测。

    例如：正向反应预测，不使用 beam search
    ```
    python d_rxn_prediction.py --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --mode 'forward' --n_beam 1 
    ```
    例如：逆反应预测，使用 k=3 的 beam search
    ```
    python d_rxn_prediction.py --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --mode 'retro' --n_beam 3 
    ```

6. 使用分子/溶剂双输入的 FluoDB 荧光性质回归

    项目已经可以扩展为支持本地 FluoDB 风格的数据划分，其中每个样本包含 `smiles`、`solvent` 以及一个目标列。当前仓库已兼容本地 `FlourDB/` 目录结构：

    * `abs_train.csv`, `abs_valid.csv`, `abs_test.csv`
    * `emi_train.csv`, `emi_valid.csv`, `emi_test.csv`
    * `plqy_train.csv`, `plqy_valid.csv`, `plqy_test.csv`
    * `e_train.csv`, `e_valid.csv`, `e_test.csv`

    示例：
    ```
    python d_fluodb_regression.py --task emi --data_dir './FlourDB' --checkpoint './Pretrain/checkpoint_SPMM.ckpt' --device cuda
    ```

    可选任务包括 `abs`、`emi`、`plqy` 和 `e`。脚本会分别编码荧光团与溶剂，再融合两者嵌入，并将最佳检查点保存到 `./output/FluoDB/`。

7. 面向已训练 SPMM 回归器的 FluoDB 原子级解释

    在训练好目标相关模型后，可以通过原子级 token masking 估计哪些荧光团原子会推动预测性质升高或降低。下面给出发射模型的解释示例：

    ```
    python d_fluodb_explain.py --targets emi --checkpoint './output/FluoDB/emi_best.pth' --fluorophore 'Cc1ccc(C(=O)c2cc(C(=O)O)cc3c2CCN3c2c(Cl)cccc2Cl)cc1' --solvent 'O' --device cpu
    ```

    若要同时解释四个目标，请先训练四个检查点，然后运行：

    ```
    python d_fluodb_explain.py --targets abs emi plqy e --model_dir './output/FluoDB' --fluorophore 'Cc1ccc(C(=O)c2cc(C(=O)O)cc3c2CCN3c2c(Cl)cccc2Cl)cc1' --solvent 'O'
    ```

    脚本会将 CSV 归因表和 PNG 原子高亮图写入 `./pred/spmm_fluodb_explain/`。

## 致谢
* `xbert.py` 与 `scheduler` 中带交叉注意力层的 BERT 代码修改自 [ALBEF](https://github.com/salesforce/ALBEF)。
* SMILES 增广代码来自 [pysmilesutils](https://github.com/MolecularAI/pysmilesutils)。
