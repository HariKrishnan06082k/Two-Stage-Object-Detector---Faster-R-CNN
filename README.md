# Two Stage Detector 

The project implements a two-stage object detector, based on [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) which consists of two modules- Region Proposal Networks (RPN) or otherwise called rpn head and Fast R-CNN. 
The network is trained to detect a set of object classes and evaluate detection accuracy using the classic metric mean AveragePrecision [mAP](https://github.com/Cartucho/mAP)
<img src="img/architecture.png" width=600>
