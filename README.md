# SEEChat 2.0 视觉多模态模型
SEEChat 2.0主要变化：
1. **目标定位**：新增目标定位能力，能根据用户的描述给出对应目标在图像中的bounding box坐标    
2. **深层融合**：模型结构从浅层融合切换为深层融合，提升模型对视觉信息的理解能力    
3. **严格超集**：LLM保持frozen，视觉能力的加入不影响内嵌的语言模型在文本任务上的原有能力    

## Contents
  - [Introduction](#introduction)
  - [Method](#method)
  - [Capability](#capability)
  - [Deploy](#deploy)
  - [Citation](#citation)
  - [References](#references)

## Introduction
SEEChat 是一个侧重视觉能力的多模态对话模型，目标包含三方面：    
1. 为文本单模态的语言模型增加对视觉信息的理解和处理能力，为LLM增加“眼睛”    
2. 基于多模态融合，实现视觉多任务的统一模型    
3. 探索视觉与文本信息的对齐、转写、及信息互补    

## Method
SEEChat 2.0仍然基于单模态专家缝合路线，而非原生多模态路线。与1.0版本的重大区别在于，2.0版的模型结构从1.0版的浅层融合切换为现在的深层融合方式。    

## Capability

## Deploy

## Citation

## References
