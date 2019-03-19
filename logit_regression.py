import numpy as np
import pandas as pd


data_original = pd.read_excel('data/3.0.xlsx')
print(data_original)
density = list(data_original[u'Density'])
suger_ratio = list(data_original[u'SugerRatio'])
label = np.ones(len(suger_ratio))
x = np.array([density, suger_ratio, label])
print(x)
y = np.array(data_original[u'Label'])
print(y)

beta = np.array([[0.0], [0.0], [1.0]])  # beta初始向量值
prior_l = 0
count = 0  # 记录迭代次数

while True:
    beta_x = np.dot(beta.T[0], x)    # beta.T = [[0,0,1]]
    print()
    log_likelihood = lambda i:(-y[i] * beta_x[i] + np.log(1 + np.exp(beta_x[i])))
    cur_l = sum(map( log_likelihood, range(17) ))

    # 判断目标函数值l知否达到最小值（0.00001的精度），一旦达到即退出while循环
    if np.abs(cur_l - prior_l) <= 0.00001:
        break

    # 牛顿法 迭代deta值
    d_beta = 0   # beta一阶导数，3.30式
    d2_beta = 0   # beta二阶导数，3.31式
    l1 = lambda i:np.dot(np.array([x[:, i]]).T, (y[i]-(np.exp(beta_x[i])/(1 + np.exp(beta_x[i])))))
    l2 = lambda i:np.dot(np.array([x[:, i]]).T, np.array([x[:, i]]).T.T)*(np.exp(beta_x[i])/(1 + np.exp(beta_x[i])))*(
                  1-(np.exp(beta_x[i])/(1 + np.exp(beta_x[i]))))

    d_beta  -= sum(map(l1, range(17)))
    d2_beta += sum(map(l2, range(17)))
    beta -= np.dot(np.linalg.inv(d2_beta), d_beta)

    count += 1
    prior_l = cur_l

print('beta={}\ncount={}'.format(beta, count))
