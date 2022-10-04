---
title: shell script 学习笔记（一）
date: 2022-10-04 22:08:30 +0800
categories: [Shell Script, Basics]
tags: [variable, regex]
---

本文主要受益于《鸟哥的 Linux 私房菜》。

shell 中有一个很有意思的变量 `PS1`，用于规定提示字符的形式，详见 [Bash PS1 customization examples](https://linuxhint.com/bash-ps1-customization/)。

## shell 变量

### 变量定义

bash 中变量分为两种：**环境变量**与自定义变量。

定义自定义变量：

```shell
$ myname=XT
```

这样就定义了一个自定义变量 `XT`，注意 **`=` 两边不能有空格！**

若要定义环境变量，则需要使用 `export` 命令将自定义变量变为环境变量：

```shell
$ export myname=XT
```

这样就定义了一个环境变量 `myname`，它的值为 `XT`。

要删除一个变量也很简单，使用 `unset` 命令就可以：

```shell
$ unset myname
```

这样就删除了变量 `myname`。

### 变量作用域的问题

这两种变量主要区别在于作用域，其中环境变量对所有子进程可见，而自定义变量则只对本进程可用。**这是因为子进程只会继承父进程的环境变量，而不会继承父进程的自定义变量。**

### 变量操作

变量的普通赋值很容易，模式为 `var=value`，其中 `value` 为常量，且会被认为是字符串。

然而当遇到需要从现有变量中构造出新的变量，下面这些模式就显得非常有用：

#### 变量截取

| 变量设置方式           | 说明                                    |
|:----------------:|:-------------------------------------:|
| ${变量#关键词}        | 若【变量】内容**从头开始**的数据符合【关键词】，则将符合的最短数据删除 |
| ${变量##关键词}       | 若【变量】内容**从头开始**的数据符合【关键词】，则将符合的最长数据删除 |
| ${变量%关键词}        | 若【变量】内容**从尾向前**的数据符合【关键词】，则将符合的最短数据删除 |
| ${变量%%关键词}       | 若【变量】内容**从尾向前**的数据符合【关键词】，则将符合的最长数据删除 |
| ${变量/旧字符串/新字符串}  | 若变量内容满足【旧字符串】则**第一个**【旧字符串】会被【新字符串】替换 |
| ${变量//旧字符串/新字符串} | 若变量内容满足【旧字符串】则**全部的**【旧字符串】会被【新字符串】替换 |

#### 变量选择性赋值方式

名字我瞎起的，大概知道什么意思就行，下面是模式：

| 变量设置方式           | str 没有设置              | str 为空字符              | str 已设置为非空字符串       |
|:----------------:|:---------------------:|:---------------------:|:-------------------:|
| var=${str-expr}  | var=expr              | var=                  | var=$str            |
| var=${str:-expr} | var=expr              | var=expr              | var=$str            |
| var=${str+expr}  | var=                  | var=expr              | var=expr            |
| var=${str:-expr} | var=                  | var=                  | var=expr            |
| var=${str=expr}  | str=expr<br/>var=expr | str=不变<br/>var=expr   | str 不变<br/>var=$str |
| var=${str:=expr} | str=expr<br/>var=expr | str=expr<br/>var=expr | str 不变<br/>var=$str |
| var=${str?expr}  | expr 输出至 stderr       | var=                  | var=$str            |
| var=${str:?expr} | expr 输出至 stderr       | expr 输出至stderr        | var=$str            |

## 输出重定向

输出重定向主要是通过 `>` 和 `>>` 将原本输出到 `stdout` 和 `stderr` 的信息输出到指定的文件中。其中 `stdout` 的文件描述符为 `1`，`stderr` 的文件描述符为 `2`，借助输出重定向我们也可以将原本都输出到终端的 `stdout` 和 `stderr` 信息分开。比如：

```shell
./something.sh 1> stdout.txt 2> stderr.txt
```

这是比较基本的操作。我写这一部分主要是为了记录一种文件描述符绑定的方式，比如我现在要将 `stderr` 的输出内容也输出到 `stdout`，可以用以下的命令：

```shell
# correct
./something.sh 1> stdout.txt 2>&1

# wrong, 会被 bash 认为将 stderr 的信息输出到文件名为 1 的文件中
./something.sh 1> stdout.txt 2> 1
```

这里多了一种特殊语法 `&`，含义为告诉 `bash`，后面跟着的不是文件名，而是一个文件描述符。详见 [StackOverflow](https://stackoverflow.com/questions/818255/what-does-21-mean)。

## 一些有用的命令

### uniq

为单词 `unique` 的简写。顾名思义，可以消除相同结果的输出：

```shell
# echo without uniq
$ echo -e "test\ntest"
> test
> test

$ echo -e "test\ntest" | uniq
> test

# 可以使用 -i 选项来忽略大小写的不同
$ echo -e "test\nTEST" | uniq -i
> test

# 同时还可以使用 -c 选项来对重复的内容进行计数
$ echo -e "test\ntest\nTEST\n" | uniq -c
> 2 test
> 1 TEST

# 但是貌似只能合并相邻
$ echo -e "test\nTEST\ntest" | uniq -c
> 1 test
> 1 TEST
> 1 test
```

由上例可以看出 `uniq` 只能相邻内容的重复，那若想要不受相邻的限制呢？可以使用 `sort` 命令先对内容进行排序。`sort` 命令的详细用法参见 [man page](https://man7.org/linux/man-pages/man1/sort.1.html)。

### tee

`tee` 命令用于双向重定向。双向重定向也就是把输出内容重定向到两个目的地。比如在运行一个程序的时候既想把结果输出到终端上，也想保存在某个文件中，这时候就可以使用 `tee` 命令：

```shell
# 输出 ls -l 结果，同时保存在 files.txt 中（覆写）
ls -l | tee files.txt

# 输出 ls -l 结果，同时保存在 files.txt 中（追加）
ls -l | tee -a files.txt
```

### xargs

`xargs` 命令用于产生某个命令的参数。是不是感觉根本不知道什么意思？我也差不多:)，也只能理解个大概使用方法，详细描述参见 [man page](https://man7.org/linux/man-pages/man1/xargs.1.html)。如果你的目的和我一样只要略知一二即可，不如直接来看例子：

```shell
# 如果我们想列出当前文件夹下的文件详细信息，我们会使用 ls -l 命令。但如果我们只想要列出权限为 644 的文件信息呢？
# 一种方法是使用管道命令，先使用 ls -l 将所有结果列出来，然后通过管道将结果输送给 grep，再由 grep 找出权限为 644 的文件再加以输出。
ls -al | grep .*rw-r--r--
# 另一种方式就是使用 xargs。可以先用 find 命令找出所有权限为 644 的文件，再使用 xargs 将这些文件变成 ls -l 指令的参数。
find . -maxdepth 1 -perm 644 | xargs ls -l
```

## bash 通配符

需要与后面的正则表达式区分开：

| 符号  | 意义                             |
|:---:|:------------------------------:|
| *   | 代表【0 个到无穷多个】任意字符               |
| ?   | 代表【一定有一个】任意字符                  |
| []  | 同样代表【一定有一个在括号内】的字符（非任意字符）      |
| [-] | 若有减号在中括号内时，代表【在**编码顺序**内的所有字符】 |
| [^] | 若中括号内第一个字符为^，就代表【反向选择】         |

## 基础正则表达式

### 特殊符号

bash 下正则表达式的特殊符号：

| 特殊符号       | 代表意义                           |
|:----------:|:------------------------------:|
| [:alnum:]  | 代表英文大小写字符及数字                   |
| [:alpha:]  | 代表英文大小写字符                      |
| [:blank:]  | 代表空格与制表符 \t                    |
| [:cntrl:]  | 代表控制按键，包括 CR、LF、Tab、Del 等      |
| [:digit:]  | 代表数字                           |
| [:graph:]  | 代表除了空格和制表符 \t 之外的所有按键          |
| [:lower:]  | 代表小写字符                         |
| [:upper:]  | 代表大写字符                         |
| [:print:]  | 代表任何可以打印出来的字符                  |
| [:punct:]  | 代表标点符号，即 ;:?!'"#$              |
| [:space:]  | 代表所有会产生空格的符号，包括 空格，制表符 \t 和 CR |
| [:xdigit:] | 代表 16 进制的符号，包括 0-9, a-f, A-F   |

### 自定义字符集合

类似于 bash 的通配符，bash 正则表达式也使用中括号 `[]` 来选择字符集合：

```shell
# 在 sample.txt 中寻找包含 tast 或 test 的行
$ grep -n 't[ea]st' sample.txt
```

同样的也有反选机制，也是用 `^` 来标识：

```shell
# 在 sample.txt 中寻找包含字符 foo，但 foo 之前没有字符 g 的行
$ grep -n '[^g]foo' sample.txt
```

### 行首行尾匹配

bash 正则表达式用于匹配行首的字符为 `^`，用于匹配行尾的字符为 `$`。

看似好像这里的 `^` 和自定义字符集合的 `^` 重复了对吧。但其实区别在于 `^` 出现的位置。若 `^` 出现在中括号内就代表反选，否则就代表匹配行首。

```shell
# 在 sample.txt 中查找以单词 the 开头的行
$ grep -n '^the' sample.txt

# 在 sample.txt 中查找以单词 the 结尾的行
$ grep -n 'the$' sample.txt

# 找出 sample.txt 中的空行
$ grep -n '^$' sample.txt
```

### 任意字符 `.` 与重复字符 `*`

bash 正则表达式中，`.` 代表**任意字符**（同时也有**一定有一个字符**的意思），而 `*` 代表**重复前一个字符，0 次或无穷次**。

这个很好理解就不给例子了。

### 限定连续字符数量

bash 正则表达式中可以使用 `{}` 来限定连续字符的数量。比如：

```shell
# 在 sample.txt 中找出存在连续的 2-5 个 O 的行
$ grep -n 'O\{2,5\}' sample.txt

# 在 sample.txt 中找出存在连续的 2 个 O 的行
$ grep -n 'O\{2\}' sample.txt

# 在 sample.txt 中找出存在大于连续的 2 个 O 的行
$ grep -n 'O\{2,\}' sample.txt
```

## 扩展正则表达式

### 查找多个模式

可以使用 `|` 符号将多个正则表达式组合成 `or` 的含义：

```shell
# 在 sample.txt 中找出以 # 开头的行以及空行
$ grep -n '^#|^$' sample.txt
```

### `+` 与 `?`

`+` 符号类似于 `*`，会匹配重复 **一个或多个以上** 的前一个字符。

`?` 符号会匹配 **0 个或 1 个** 前一个字符。

### 群组功能

使用 `()` 实现群组功能。比如我想查找包含字符串 glad 和 good 的行：

```shell
$ grep -n 'g(la|oo)d' samples.txt
```

类似于普通的单个字符，群组后面也可以加 `+, *, ?, {}` 等数量限定符。

