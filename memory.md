当前项目，在 AnySplat 的 3D Gaussian Splatting 编解码框架上，移植了 IGGT 的 instance head（SamProjector + PartHead + 对应 loss），并通过新的自定义 dataset 与 instseg DataModule，对同一网络进行额外训练，以同时完成 3DGS 重建和实例级分割。
基于AnySplat预训练权重进行训练，重建相关的网络已经优化过了，只有instance head等新增的结构是随机初始化的。
IGGT官方实现本身并未开源训练部分，所以loss的计算是民间实现，可能有误。