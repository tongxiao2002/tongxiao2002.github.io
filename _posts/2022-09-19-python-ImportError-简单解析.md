---
title: python ImportError 简单解析
date: 2022-09-19 13:12:30 +0800
categories: []
tags: [python]
---

本文主要受益于[StackOverflow](https://stackoverflow.com/questions/16981921/relative-imports-in-python-3).

## 起因

写这篇文章的主要起因是曾经用 `python` 写深度学习的时候将文件组织成了类似如下形式：

```python
src
├── trainer/
│   ├── __init__.py
│   ├── train.py
├── utils/
│   ├── __init__.py
│   ├── utils.py
```

在 `train.py` 中 `import` 了 `utils.py`，这在代码编写时 vscode 都能正常解析出来，也没有给出语法报错。但是运行时 `python` 就一直各种报错:
```python
ModuleNotFoundError: No module named 'src'

ImportError: attempted relative import with no known parent package
```
试便了所有 import 方式都不行。

然后昨天同学把我叫去帮忙 debug，又遇到了这类似的问题，因此回来后花了一晚上仔细阅读了很多资料，自认为有了一定了解，所以记录一下。

## python 的找包路径 —— `sys.path`

当我们在使用 `import [Module]` 时，`python` 会在 `sys.path` 保存的路径中去寻找这个 `[Module]`。如[官方文档](https://docs.python.org/3/library/sys.html#sys.path)中对 `sys.path` 的描述：
> A list of strings that specifies the search path for modules. Initialized from the environment variable `PYTHONPATH`, plus an **installation-dependent default**.

因此我们出现 `ModuleNotFoundError` 多半也是因为这个变量。

## 为什么 `sys.path` 会导致这些问题？

正如[官方文档](https://docs.python.org/3/library/sys.html#sys.path)所说：
> Initialized from the environment variable `PYTHONPATH`, plus an **installation-dependent default**.

这里 `installation-dependent default` 是什么东西我暂且还不清楚，但是 `PYTHONPATH` 这个变量一看就感觉和 `site-packages` 相关。

可以打开 `python` 解释器，终端输入：

```python
>>> import sys
>>> sys.path
['',
'*/anaconda3/envs/main/lib/python38.zip',
'*/anaconda3/envs/main/lib/python3.8',
'*/anaconda3/envs/main/lib/python3.8/lib-dynload',
'*/anaconda3/envs/main/lib/python3.8/site-packages']
```

看到这你可能就要问了：我寻思这几个路径和我们自己的包什么关系都没有，怎么说这个 `sys.path` 和我们之前的报错有关？

注意到上面 `sys.path` 第一个值为空，其实这个就是关键。我们继续来看[官方文档](https://docs.python.org/3/library/sys.html#sys.path)中对 `sys.path` 的描述：
> As initialized upon program startup, <u>the first item of this list, path[0], is the directory containing the script that was used to invoke the Python interpreter.</u> If the script directory is not available (e.g. if the interpreter is invoked interactively or if the script is read from standard input), path[0] is the empty string, which directs Python to search modules in the current directory first.

上例中为空其实只是因为我们是使用终端去启动 `python` 的，因此 `sys.path` 的第一个值会被置为空。

在下例中
```python
src
├── trainer/
│   ├── __init__.py
│   ├── train.py
├── utils/
│   ├── __init__.py
│   ├── utils.py
```

假设我们工作目录为 `src`，如果我们使用诸如 `python trainer/train.py` 来启动 `python`，那么此时 `sys.path` 的第一个值将会是 `/path/to/src/trainer`。

所以这就造成了一个误解，我们以为 `python` 的搜索路径将会和我们的工作目录一样是 `src`，但是实际上 `python` 的搜索路径是 `src/trainer`。

因此如果我们在 `train.py` 中使用如下方式导入 `utils`，会导致找不到对应的包：

```python
from utils import utils
```

因为 `utils` 这个包不存在于搜索路径 `/path/to/src/trainer` 之下。

而这个 `python` 的搜索路径又有点类似于 linux 中的 `/`，不能再上一层，参见[PEP328](https://peps.python.org/pep-0328/#relative-imports-and-name)
> Relative imports use a module’s __name__ attribute to determine that module’s position in the package hierarchy. <u>If the module’s name does not contain any package information (e.g. it is set to ‘__main__’) then relative imports are resolved as if the module were a top level module, regardless of where the module is actually located on the file system.</u>

其中 `Top Level Module` 就有点类似 linux 的 `/`。当然只是个人理解，可能存在错误。

这就导致了尽管我们理解了搜索路径是 `src/trainer`，然后我们在 `train.py` 中使用如下方式导入 `utils`：

```python
from ..utils import utils
```

还是会引发 `ImportError: attempted relative import with no known parent package`，因为无法再向上一层。

看到这里，我们解决这个问题的方法核心就在于改变 `sys.path` 第一项的值，使得它指向 `/path/to/src`。

## 解决方法

### 改变文件组织结构

个人认为最好的方式应该为改变自己的文件组织结构。用于执行的脚本 `train.py` 应当位于整个项目的 `Top Level`：

```python
src
├── train.py
├── utils/
│   ├── __init__.py
│   ├── utils.py
```

这样在执行 `python train.py` 时，`sys.path` 就会被设置成 `/path/to/src`，因此 `utils` 包也就会处于 `python` 的搜索路径之下，问题得到解决。

### 手动改变 `sys.path` 的值

直接修改 `sys.path` 可以说是直捣黄龙，管你花里胡哨的我直接把你源头改了。但是这种直接更改重要变量的方式，我个人认为是治标不治本，且不说需要在所有可能出现问题的地方都对 `sys.path` 进行修改，而且可能会存在一些副作用（猜测，不确定）。可以在测试或者自己的玩具项目中用一用，如果对于正在开发一个大项目或者对代码优雅性要求较高的同学建议还是别用了。

同样在上例中，我们在 `train.py` 的最开头加上这么几句：

```python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
```

这就将 `/path/to/src` 加入到了 `sys.path` 中。

### 使用 `python -m` 执行

这种方式本质上也是更改了 `sys.path` 的第一项。参考[官方文档](https://docs.python.org/3/using/cmdline.html#cmdoption-m)：
> If this option is given, the first element of sys.argv will be the full path to the module file (while the module file is being located, the first element will be set to "-m"). As with the -c option, <u>the current directory will be added to the start of sys.path</u>.

也就是说如果我在 `/path/to/src` 中执行了 `python -m src.trainer.train`，那么我的 `sys.path` 第一项仍然会是 `/path/to/src`。问题得到解决。

### 使用 `setuptools` 安装之后再导入

这种方式的具体原理我也不太清楚，来源于[StackOverflow](https://stackoverflow.com/questions/16981921/relative-imports-in-python-3)。

具体方式为，首先将文件组织成如下结构：

```python
project
├── src/
│   ├── train.py
│   ├── utils/
│   ├── __init__.py
│   ├── utils.py
├── setup.py
```
其中 `setpy.py` 内容为：

```python
from setuptools import setup, find_packages
setup(
    name = 'src',
    packages = find_packages(),
)
```

此时若还用以上 `sys.path` 的理论（可能已经与 `sys.path` 无关了，方便类比就再拿来说一下），此时 `python` 的搜索路径为 `/path/to/project` 从而在 `train.py` 中导入 `utils.py` 的方式为：

```python
from project.src.utils import utils
```

然后安装 `project`，再执行：

```shell
~$ cd project/
~/project$ python3 setup.py install --user
~/project$ python3 project/src/trainer/train.py
```

这种方式应该只适用于需要发布的项目，如果是自己简单写的一些应用使用这种方式来解决 `ImportError` 可能性不大。

我个人之前在对其中原理不清楚的时候也被这种先安装再 `import` 的方式所误导过，所以写下这种解决办法也相当于是做个提醒。
