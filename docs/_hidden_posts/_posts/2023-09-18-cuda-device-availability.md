# Cuda Device Availability

{% highlight shell %}
nvidia-smi
{% endhighlight %}

{% highlight shell %}
CUDA_VISIBLE_DEVICES=0,2,3 python main.py
{% endhighlight %}

{% highlight python %}
DEVICE = 'cuda' # 'cuda:0'
{% endhighlight %}