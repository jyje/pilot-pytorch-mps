# pilot-pytorch-mps

pilot setup for pytorch with mps (but including cuda, cpu also)


## Getting Started

### Setup environment

Install the required packages using the following command:
```bash
cd src
pip install -r requirements.txt
```

Run
```bash
python main.py
```

If it runs without any error, you are good to go.
```bash
MPS is available and built. Using Apple Silicon GPU.
0 76.33473205566406
100 21.244474411010742
200 6.49550724029541
300 2.5242221355438232
400 1.4398127794265747
...
4500 0.9754587411880493
4600 0.9754587411880493
4700 0.9754587411880493
4800 0.9754588007926941
4900 0.9754588007926941
Final loss: 0.9754588007926941
Result: y = 2.981515884399414x + -0.038626547902822495
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
