# CovNet-UFCSPA

## Abstract
The prompt and proper identifying COVID-19 in patients provides adequate screening, thus preventing the worsening of symptoms caused by the disease. This research proposes the CovNet-UFCSPA architecture comprising data from clinical images (X-ray) pre-processing and DML. A total of 24 235 images for model training, validation, and testing. Clipping techniques, GCE and CLAHE  in the pre-processing of the images. The architecture had a 99\% recall when used to classify x-rays from Brazilian patients at Hospital da Clínica de Porto Alegre of the Federal University of Rio Grande do Sul (HCPA - UFRGS). Applying CLAHE and removing the X-ray region of interest improved the FN rate, decreasing the model classification results from 187 to 9. In addition, the architecture provided a tool that can be useful to health professionals with a score metric and the heat map of the tested images. For performance analysis, the architecture is compared to a Resnet50 V2 and an Inception V3; the results showed that the CovNet-UFCSPA architecture obtained the best FN and TP rates and recall.


Open Data Set
 Outer pipes  Cell padding 
No sorting
| Referências                                                                | Composição do dataset                                                      | Técnica de IA                                                                          | Tipo

Imagem | Link para o dataset ou código fonte                                                                                                                                                                                                                |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [(JAIN et al., 2020)](https://www.zotero.org/google-docs/?foNnLC)          | Normal: 1373 

Pneumonia viral: 1 493 

Pneumonia bacteriana: 2 780 

<br> | ResNet50 e ResNet101: para

classificação \*Rede pré-treinada pelo dataset ImagemNet   | Raio x       | Dataset: [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) [https://github.com/ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset) |
| [(DANSANA et al., 2020)](https://www.zotero.org/google-docs/?yL6UPa)       | Pneumonia viral: 38 

COVID-19: 468 

Pneumonia bacteriana: 48             | VGG-16, Inception-V2 e Árvore de decisão: para classificação                           | Raio x       | Dataset

[https://github.com/ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)                                                                                                                                |
| [(ANGELOV; SOARES, 2020)](https://www.zotero.org/google-docs/?nS8SDt)      | Normal: 1 252

COVID-19: 1 229 

<br>                                      | Resnet, VGG-16, GoogleNet,

Árvore de decisão, AlexNet, AdaBoost.                      | Tomografia   | Dataset

[https://github.com/Plamen-Eduardo/xDNN-SARS-CoV-2-CT-Scan](https://github.com/Plamen-Eduardo/xDNN-SARS-CoV-2-CT-Scan)                                                                                                                    |
| [(RAIKOTE, 2020)](https://www.zotero.org/google-docs/?Q0QT5l)              | Normal: 90

COVID-19: 137 

Pneumonia viral: 90 

<br><br>                 | \-                                                                                     | Raio x       | Dataset

[https://www.kaggle.com/pranavraikokte/covid19-image-dataset](https://www.kaggle.com/pranavraikokte/covid19-image-dataset)                                                                                                                |
| [(ANAS M. TAHIR et al., 2022)](https://www.zotero.org/google-docs/?SqGATD) | Normal:10 701

COVID-19:11 956 

Pneumonia não COVID-19: 1 263

<br>       | \-                                                                                     | Raio x       | Dataset:

[https://www.kaggle.com/anasmohammedtahir/covidqu](https://www.kaggle.com/anasmohammedtahir/covidqu)                                                                                                                                     |
| [(RAHAMAN, 2021)](https://www.zotero.org/google-docs/?I2ZDZy)              | COVID-19: 2616

Viral: 1345

<br>                                          | \-                                                                                     | Raio x       | Dataset

[https://www.kaggle.com/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)                                                                                                  |
| [(YANG et al., 2020)](https://www.zotero.org/google-docs/?cKnMau)          | COVID-19: 349 + 589

Normal: 463                                           | Resnest-50 e DenseNet-169                                                              | Tomografia   | DataSet

https://paperswithcode.com/dataset/covid-ct

[https://github.com/ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)                                                                                   |
| [(ZHANG et al., 2020)](https://www.zotero.org/google-docs/?pIPxha)         | COVID-19: 1 578

Pneumonia não COVID-19:

1614

Normal 1 164               | DeepLabv3: segmentação do pulmão. 3D ResNet: para classificação                        | Tomografia   | Deeplabv3: [https://github.com/pytorch/vision](https://github.com/pytorch/vision)

<br>

Código e Dataset: [http://ncov-ai.big.ac.cn/download?lang=en](http://ncov-ai.big.ac.cn/download?lang=en)                                                  |
| [(BAI et al., 2020)](https://www.zotero.org/google-docs/?bS3TvO)           | COVID-19: 521

Pneumonia não COVID-19: 665                                 | EfficienteNets B4: para classificação.                                                 | Tomografia   | [https://github.com/robinwang08/COVID19](https://github.com/robinwang08/COVID19)                                                                                                                                                                   |
| [(HARMON et al., 2020)](https://www.zotero.org/google-docs/?Bv6sx5)        | COVID-19: 1 029

Não COVID-19:  1695

<br>                                 | SDK disponibilizada pela NVIDIA: para classificação AH-Net: para segmentação do pulmão | Tomografia   | AH-Net: [https://github.com/lsqshr/AH-Net](https://github.com/lsqshr/AH-Net)

DataSet: [https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)                            |
| [(VAYÁ et al., 2020)](https://www.zotero.org/google-docs/?4g9JmJ)          | COVID-19: 3 642

Não COVID-19:  2 120                                      | *   <br>                                                                               | Raio x       | dataset: 

[https://paperswithcode.com/dataset/padchest](https://paperswithcode.com/dataset/padchest)

<br>

[https://b2drop.bsc.es/index.php/s/BIMCV-COVID19](https://b2drop.bsc.es/index.php/s/BIMCV-COVID19)                                    |
📋 Copy
Clear
Buy Me a Coffee at ko-fi.com
