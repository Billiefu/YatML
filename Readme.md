# Machine Learning & Data Mining [SYSU CSE 2024-1]

> Copyright © 2024 Fu Tszkok.

## Repository Description

This repository contains all assignments and corresponding implementation code for the **Machine Learning and Data Mining** course offered at Sun Yat-sen University during the 2024-1 semester. The course is taught by Associate Professor Liang Shangsong, who employs a step-by-step teaching approach to guide students from foundational concepts and principles of machine learning to cutting-edge research on advanced models in the field. While many of these models fall under the domain of deep learning, they fundamentally remain within the broader scope of machine learning. Out of respect for and to protect the copyright of course materials, this repository does not include any lecture slides (PPT) or related resources. For those interested in exploring the assignments and implementations from Associate Professor Su Qinliang's Machine Learning and Data Mining course, please refer to the repository maintained by Ma Dai, which contains necessary reference materials. ([KDAIer/machine-learning: machine\_learning (github.com)](https://github.com/KDAIer/machine-learning))

## Copyright Statement

All original code in this repository is licensed under the **[GNU Affero General Public License v3](LICENSE)**, with additional usage restrictions specified in the **[Supplementary Terms](ADDITIONAL_TERMS.md)**. Users must expressly comply with the following conditions:

* **Commercial Use Restriction**
  Any form of commercial use, integration, or distribution requires prior written permission.
* **Academic Citation Requirement**When referenced in research or teaching, proper attribution must include:

  * Original author credit
  * Link to this repository
* **Academic Integrity Clause**
  Prohibits submitting this code (or derivatives) as personal academic work without explicit authorization.

The full legal text is available in the aforementioned license documents. Usage of this repository constitutes acceptance of these terms.

## Repository Contents

* Homework1 - Exercise for Monte Carlo Methods
* Homework2 - Evaluation Metrics
* Homework3 - Linear Regression
* Homework4 - SVM

Additionally, in the root directory of this repository, you will find the repository's documentation (`Readme.md`), its environment configuration file (`environment.yml`), and the open-source license (`LICENSE`).

## Environment

To run the code in this repository, you need to set up the required environment. Although configuring a Conda environment is not particularly difficult, this repository provides a pre-defined environment setup stored in the `environment.yml` file located in the root directory.

To configure the environment, ensure that you have Anaconda or Miniconda installed. After installing the environment, switch to the current directory in the command line (cmd) and run the following command:

```shell
conda env create -f environment.yml
```

If there are no issues, the environment should be created successfully. You can then activate the environment by running the following command in the command line:

```shell
conda activate YatML
```

If you are using PyCharm, VSCode, or other IDEs, you can configure the environment directly within the IDE and run the relevant programs from there.

## Acknowledgments

I sincerely extend my gratitude to Associate Professor *Liang Shangsong* for his expert guidance in the field of machine learning, which has been pivotal in shaping my academic journey. I am equally thankful to my fiancée, Ms. *Ma Yujie*, for her unwavering support and quiet encouragement. Additionally, I would like to express my heartfelt appreciation to classmates *Ma Dai* and *Wang Chenyu*, whose invaluable insights and timely assistance were instrumental during critical phases of the code implementation.

## Contact & Authorization

For technical inquiries, academic collaboration, or commercial licensing, contact the copyright holder via:

* **Academic Email**: `futk@mail2.sysu.edu.cn`
* **Project Discussions**: [Github Issues](https://github.com/Billiefu/YatML/issues)
