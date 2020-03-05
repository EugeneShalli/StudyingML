import scipy as sp
import matplotlib.pyplot as plt

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
print(data[:10])
print(data.shape)

x = data[:, 0]
y = data[:, 1]

print(sp.sum(sp.isnan(y)))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

print(sp.sum(sp.isnan(y)))

plt.scatter(x, y, s=10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
           ['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
plt.grid(True, color='0.75')
#plt.show()

def error(f, x, y):
    return sp.sum((f(x) - y)**2)

# Аппроксимация данных (поиск подходящей вункции, где 1 - степень многочлена)
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)

print("Параметры модели: %s" % fp1)

f1 = sp.poly1d(fp1)

# Абсолютная величина погрешности
print(error(f1, x, y))

fx = sp.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), linewidth=4, color="green")
plt.legend(["d=%i" % f1.order], loc="upper left")
plt.show()

f2 = sp.polyfit(x, y, 2)



