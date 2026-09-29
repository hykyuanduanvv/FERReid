# 代码来源与第三方说明

## VICP

`models.py`、`custom_trainer.py`、`ops/` 来自服务器现有 VICP 派生实现，包含本地加载、精度与 Trainer 接口兼容修改。原工作：Huang 与 Liu，*VICP: Generalizable Object Re-Identification via Visual In-Context Prompting*，ICCV 2025，https://arxiv.org/abs/2508.21222 。

本次可取得的本地 VICP 副本没有独立 LICENSE 文件；未据此推断它为 MIT，也未擅自给整仓库添加统一许可证。原文件中的作者/版权注释保留，例如 `custom_trainer.py` 的 Nabarun Goswami (2024) 注释、`ops/losses.py` 的 Yonglong Tian 注释。更大范围的公开分发与再许可应核对各上游授权；私有研究整理不改变原版权归属。

## deep-person-reid / torchreid

依赖项目：https://github.com/KaiyangZhou/deep-person-reid 。实际使用提交：`f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`。服务器该项目附带 MIT License，Copyright (c) 2018 Kaiyang Zhou。本仓库通过单独克隆依赖使用，不复制整份上游代码；使用者应保留依赖仓库中的 LICENSE。

## timm、Qwen 与数据集

ViT 从 timm 的具名预训练模型准备，Qwen3-0.6B 从其模型发布库下载。库代码许可、预训练权重许可和数据集使用条款是独立事项；本仓库不分发这些权重和数据图像，也不替代它们的许可。

原代码备份来源为服务器 `/root/FERReID` 工作树。该树基于 scaffold commit，但包含未提交修改，因此单靠旧 Git commit 无法定位全部实验代码；文件级来源 SHA256 已保留在 `results/archive_20260929/provenance.json`。
