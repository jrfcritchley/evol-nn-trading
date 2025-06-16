# evol-nn-trading
A system for training neural net agents to trade stocks using an evolutionary algorithm.

![Figure_2_AvgFinBal](https://github.com/user-attachments/assets/5e91f3ca-253f-438c-9605-88d721ff42d1)

This is notably a poor stratedgy for learning stock trading as fitness fluctuates greatly
from one generation to another, this makes it difficult to learn without losing large quantites
of balance/fitness. The graph above shows an example of a simulation that took 439 generations before an
agent broke even and this strategy is not even maintained.

To fill the Stocks folder use Yahoo Finance to download the past 30 years or so of NASDAQ 
or NYSE prices, the example folder is just to illustrate what it should look like and training
on the example will not be efficacious.
