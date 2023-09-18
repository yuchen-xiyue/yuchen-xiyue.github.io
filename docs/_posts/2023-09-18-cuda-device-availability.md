# Cuda Device Availability

{% highlight shell %}
nvidia-smi
{% end highlight %}

{% highlight shell %}
CUDA_VISIBLE_DEVICES=0,2,3 python main.py
{% end highlight %}

{% highlight python %}
DEVICE = 'cuda' # 'cuda:0'
{% end highlight %}