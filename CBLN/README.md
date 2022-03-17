# Contiual Bayesian Learning Networks (CBLN)

CBLN merges different sub distribution corresponding to different tasks into a mixture gaussian distribution and leverages uncertainty mechanism to give a prediction in the testing state. e.g. The figures from left to right represent Bayesian Neural Network for Task A, Task B, and a CBLN. The distributions in the sub Bayesian Neural Networks are merged into CBLN.

<div>
<img src=images/CBLN1.png width=250 align=left alt='Task A'>
<img src=images/CBLN2.png width=250 align=center alt='Task B'>
<img src=images/CBLN_process.png width=250 align=right alt='Task A+B'>
</div>

Dependencies:
- tensorflow: 1.13.1
- tensorflow_probability: 0.6.0
- sonnet: 1.29
- keras: 2.3.0

To run the experiemnt:
`python split_mnist.py`
