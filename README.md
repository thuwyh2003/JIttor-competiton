# JStyleID

## Installation

### Prepare Environment

```text
conda create -n jdiffusion python=3.9
conda activate jdiffusion
```

### Install Requirement

```text
git clone
pip install -r requirement.txt
```

### Install JDiffusion

```text
cd JDiffusion
pip install -e .
```

### Usage

1. First, run the baseline. Instruction [here](https://github.com/JittorRepos/JDiffusion/tree/master/examples/dreambooth).
2. Second, run JStyleID（我们提交的作业里面，已经把目录都整理好了，助教批阅的时候只需要跑下面的指令就行了）

    ```text
    bash styleid.sh
    ```

    - Structure of the folder `./data`:

        ```text
        data
         |____input
         |     |____content
         |     |     |____00
         |     |     |     |____xxx.png
         |     |     |     |____yyy.png
         |     |     |     |____...
         |     |     |____01
         |     |     |____02
         |     |     |____...
         |     |____style
         |           |____00.png
         |           |____01.png
         |           |____03.png
         |           |____...
         |____output
               |____00
               |     |____xxx.png
               |     |____yyy.png
               |     |____...
               |____01
               |____02
               |____...
        ```
