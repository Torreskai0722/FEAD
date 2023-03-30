# Fuel Rate Prediction for Heavy-Duty Trucks

This repository contains the open-source dataset used in the paper:

> Liangkai Liu, Wei Li, Dawei Wang, Yi Wu, Ruigang Yang, and Weisong Shi, Fuel Rate Prediction for Heavy-Duty Trucks, accepted to IEEE Transactions on Intelligent Transportation Systems, March 2023.

The dataset includes sensor data from heavy-duty trucks, which can be used for fuel rate prediction and other research purposes.

## Dataset Description

The dataset consists of the following data:

- GPS data (latitude, longitude, and speed)
- Engine parameters (RPM, coolant temperature, intake manifold pressure, and more)
- Environmental data (temperature, humidity, and wind speed)
- Vehicle data (weight, type, and model)

The data is collected from multiple heavy-duty trucks during different driving conditions and environments.

| **Name**        | **Time** | **Trucks** | **Rows**   | **Features**                                                                   | **Download** |
| :-------------: | :------: | :--------: | :--------: | :-----------------------------------------------------------------------------: | :----------: |
| EMS dataset-1 | 12/2019  | 9          | 10,273,969 | EMS engine data, latitude, longitude, triggertime, city, road level, etc. | [link](https://drive.google.com/file/d/1lLeACfOWyvT_LtFnrN9viqU3SNu_eTBo/view?usp=sharing)         |
| EMS dataset-2 | 04/2020  | 29         | 26,145,539 | EMS engine data, latitude, longitude, triggertime, city, road level, etc. | [link](https://drive.google.com/file/d/15Awwj26PE0MIGRUW1CiCO2xkP3j8-z2o/view?usp=sharing)         |
| IFM dataset   | 06/2020  | 1          | 872,844    | IFM engine data                                                                 | [link](https://drive.google.com/drive/folders/1QOVmUDhngrTn10gaT0M31XoPlnpKARX6?usp=sharing)         |

## Citation

If you use this dataset in your research, please cite the following paper:

### BibTeX

```bibtex
@article{liu2023fuel,
  title={Fuel Rate Prediction for Heavy-Duty Trucks},
  author={Liu, Liangkai and Li, Wei and Wang, Dawei and Wu, Yi and Yang, Ruigang and Shi, Weisong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  month={March}
}