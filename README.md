# Simple app that computes FLOPs and tensor sizes of a ONNX network
## Requirements

```
onnx==1.10.1
rich==10.11.0
```

Installation:
```shell
pip3 install -r requirements.txt
```

## Usage
```
⇒  python3 main.py -h
usage: main.py [-h] -m MODEL

Compute FLOPS of operators and sizes of tensors for a ONNX model.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        ONNX model to parse
```

## Limitations
* Limited support of `MobileNet-v2`
* Supported ops: `Conv2D`, `Clip` and `Add`

## Sample output
```
⇒  python3 main.py -m mobilenetv2-7.onnx
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Operator             ┃ FLOP       ┃ Bytes IN ┃ Bytes OUT ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩
│ Conv_0               │ 43352064   │ 151424   │ 401408    │
│ Clip_1               │ 401408     │ 401408   │ 401408    │
│ Conv_2               │ 115605504  │ 401728   │ 401408    │
│ Clip_3               │ 401408     │ 401408   │ 401408    │
│ Conv_4               │ 6422528    │ 401936   │ 200704    │
│ Conv_5               │ 19267584   │ 202336   │ 1204224   │
│ Clip_6               │ 1204224    │ 1204224  │ 1204224   │
│ Conv_7               │ 1040449536 │ 1205184  │ 301056    │
│ Clip_8               │ 301056     │ 301056   │ 301056    │
(...)
```