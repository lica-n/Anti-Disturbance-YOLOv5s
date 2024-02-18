# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:18:07 2023

@author: WXY
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hyst(force, disp, du, stiff, props, statev, cmname):
    if cmname[:7] == 'UHYST01':
        Et, s, statev = khyst01(props, force, disp, du, stiff)
    elif cmname[:7] == 'UHYST02':
        Et, s, statev = khyst02(props, force, disp, du, stiff, statev)
    elif cmname[:7] == 'UHYST03':
        Et, s, statev = khyst03(props, force, disp, du, stiff, statev)
    elif cmname[:7] == 'UHYST04':
        Et, s, statev = khyst04(props, force, disp, du, stiff)
    elif cmname[:7] == 'UHYST05':
        Et, s, statev = khyst05(props, force, disp, du, stiff)
    return Et, s, statev


# 双线性滞回模型
def khyst01(props, s, e, de, Et):
    E0, sy, eta = props  # 初始刚度，屈服承载力，屈服后刚度系数
    # 胡克定律
    s = s + E0 * de  # 力
    Et = E0  # 切线刚度
    # 正向加载
    if de >= 0.0:
        # 包络线上的力
        evs = sy + (e + de - sy / E0) * eta * E0
        # 包络线上的切线刚度
        evE = eta * E0
        # 当力超过包络线时，取包络线上的值
        if s >= evs:
            s = evs
            Et = evE
    # 负向加载
    elif de < 0.0:
        evs = -sy + (e + de + sy / E0) * eta * E0
        evE = eta * E0
        if s <= evs:
            s = evs
            Et = evE
    statev = np.zeros(7)  # dummy
    return Et, s, statev


def khyst02(props, s, e, de, Et, statev):
    E0, sy, eta = props
    emax, emin, ert, srt, erc, src, kon = statev

    if kon == 0:
        emax = sy / E0
        emin = -sy / E0
        if de >= 0.0:
            kon = 1
        else:
            kon = 2
    elif kon == 1 and de < 0.0:
        kon = 2
        if s > 0.0:
            erc = e
            src = s
        if e > emax:
            emax = e
    elif kon == 2 and de > 0.0:
        kon = 1
        if s < 0.0:
            ert = e
            srt = s
        if e < emin:
            emin = e

    s = s + E0 * de
    Et = E0

    if de >= 0.0:
        evs = sy + (e + de - sy / E0) * eta * E0
        evE = eta * E0
        if s >= evs:
            s = evs
            Et = evE

        smax = max(sy, sy + (emax - sy / E0) * eta * E0)
        sres = 0.0
        eres = ert - (srt - sres) / E0
        if eres <= emax - smax / E0:
            srel = (e + de - eres) / (emax - eres) * (smax - sres) + sres
            if s > srel:
                s = srel
                Et = (smax - sres) / (emax - eres)
    elif de < 0.0:
        evs = -sy + (e + de + sy / E0) * eta * E0
        evE = eta * E0
        if s <= evs:
            s = evs
            Et = evE

        smin = min(-sy, -sy + (emin + sy / E0) * eta * E0)
        sres = 0.0
        eres = erc - (src - sres) / E0
        if eres >= emin - smin / E0:
            srel = (e + de - eres) / (emin - eres) * (smin - sres) + sres
            if s < srel:
                s = srel
                Et = (smin - sres) / (emin - eres)
    statev = emax, emin, ert, srt, erc, src, kon
    return Et, s, statev


def khyst03(props, s, e, de, Et, statev):
    E0, sy, eta = props
    ert, srt, erc, src, kon = statev[2: 7]

    emax = 0.01 * sy / E0
    emin = -0.01 * sy / E0

    if kon == 0:
        if de >= 0.0:
            kon = 1
        else:
            kon = 2
    elif kon == 1 and de < 0.0:
        kon = 2
        if s > 0.0:
            erc = e
            src = s
    elif kon == 2 and de > 0.0:
        kon = 1
        if s < 0.0:
            ert = e
            srt = s

    s = s + E0 * de
    Et = E0

    if de >= 0.0:
        evs = sy + (e + de - sy / E0) * eta * E0
        evE = eta * E0
        if s >= evs:
            s = evs
            Et = evE

        smax = 0.01 * sy
        sres = 0.0
        eres = ert - (srt - sres) / E0
        if eres <= emax - smax / E0:
            srel = (e + de - eres) / (emax - eres) * (smax - sres) + sres
            if s > srel and e + de < emax:
                s = srel
                Et = (smax - sres) / (emax - eres)
    elif de < 0.0:
        evs = -sy + (e + de + sy / E0) * eta * E0
        evE = eta * E0
        if s <= evs:
            s = evs
            Et = evE

        smin = -0.01 * sy
        sres = 0.0
        eres = erc - (src - sres) / E0
        if eres >= emin - smin / E0:
            srel = (e + de - eres) / (emin - eres) * (smin - sres) + sres
            if s < srel and e + de > emin:
                s = srel
                Et = (smin - sres) / (emin - eres)
    statev = statev = emax, emin, ert, srt, erc, src, kon
    return Et, s, statev


def khyst04(props, s, e, de, Et):
    E0, sy, eta = props

    if s * de >= 0.0:
        s = E0 * (e + de)
        Et = E0
        evs = sy + (abs(e + de) - sy / E0) * eta * E0
        evE = eta * E0
        if abs(s) >= evs:
            s = np.sign(e + de) * evs
            Et = np.sign(e + de) * evE
    elif s * de < 0.0 and e != 0.0:
        Et = s / e
        s = s + Et * de
    else:
        Et = E0
        s = 0.0
    statev = np.zeros(7)
    return Et, s, statev


def khyst05(props, s, e, de, Et):
    E0, sy, eta = props

    if abs(e + de) <= sy / E0:
        s = E0 * (e + de)
        Et = E0
    else:
        s = np.sign(e + de) * (sy + (abs(e + de) - sy / E0) * eta * E0)
        Et = eta * E0
    statev = np.zeros(7)
    return Et, s, statev


def get_peer_data(filename):
    # 输入文件路径和文件名，得到时间，地震动数据，时间步长和数据点数
    # 读取文件
    peer_file = open(filename, 'r', encoding='utf-8')
    # 创建存储时间和地震动的列表
    time_data = []
    wave_data = []
    dt = []
    # 跳过前三行
    line = peer_file.readline()
    line = peer_file.readline()
    line = peer_file.readline()
    line = peer_file.readline()
    result = re.split(r'\s', line)
    # 读取时间步长
    for i in range(len(result)):
        if result[i] != '':
            dt.append(result[i])
    dt = float(dt[5])
    # 读取接下来的数据
    while True:
        line = peer_file.readline()
        # 最后一行停止
        if not line:
            break
        result_2 = re.split(r'\s', line)
        for i in range(len(result_2)):
            if result_2[i] != '':
                wave_data.append(float(result_2[i]))
    peer_file.close()
    # 从文件中读取时间序列，根据时间步长和数据点数生成
    time_data = np.arange(0, dt * (len(wave_data) + 1), dt)
    wave_data = np.array(wave_data)
    return time_data, wave_data, dt, len(wave_data)


def response(ag, dt, npts, T, m, c, k, props, cmname):
    # 初始化响应为零
    acc = np.zeros(npts + 1)  # 加速度
    vel = np.zeros(npts + 1)  # 速度
    disp = np.zeros(npts + 1)  # 位移
    # 初始化中间变量
    statev = np.zeros(7)
    force = np.zeros(npts + 1)
    Amax, Vmax, Dmax, kt, ke, du, u_trial, du_trial, force_c, force_p, Res, df = np.zeros(12)
    # 定义Newmark方法的参数
    beta = 0.25  # 1/4 for average constant acceleration method
    gamma = 0.5  # 1/2 for average constant acceleration method
    # 定义迭代方法的参数
    tol = 1e-3  # tolerance for convergence
    max_iter = 200  # maximum number of iterations
    # 计算初始反应
    disp[0] = 0
    vel[0] = 0
    acc[0] = (-ag[0] * m - c * vel[0] - k * disp[0]) / m
    # 迭代计算
    for i in range(npts):
        # 外荷载增量
        dp = -(ag[i + 1] - ag[i]) * m
        # 等效荷载增量
        #
        dpe = ke * dp
        #
        # 试算位移
        u_trial = disp[i]
        # 起步
        if i == 0:
            kt = k  # 初始刚度
        else:
            force_p = force[i - 1]
            # 调用滞回模型，得到当前力及其导数（切线刚度）
            kt, force_c, statev = hyst(force_p, disp[i - 1], du, kt, props, statev, cmname)
        # 等效刚度
        #
        ke = kt + (m / (beta * dt ** 2)) + (c * gamma / (beta * dt))
        #
        Res = dpe - force_c  # 初始残差
        du = 0.0
        # 当前力（Current force）
        force_c = force[i]
        for j in range(max_iter):
            #########################################
            # 请在此利用牛顿法或修正牛顿法进行迭代求解du
            du_trial = Res / ke
            du += du_trial
            force_c = force_c + kt * du_trial
            Res = dpe - force_c
            if du_trial / du > tol:
                #force_c = force_c + kt * du_trial
                #Res = dpe - force_c
                continue


            else:
                break
                print(T, "Not converged - error=", du_trial / du)
        #########################################
        # 请在此利用Newmark-beta法根据du计算速度和加速度增量
        dv = du / (beta * dt) - ((gamma / beta) - 1) * vel[i] - (gamma / (2 * beta) - 1) * dt * acc[i]
        da = 1 / (beta * dt ** 2) * du - 1 / (beta * dt) * vel[i] - (1 / (2 * beta) - 1) * acc[i]
        #########################################
        # 更新位移、速度、加速度和恢复力
        disp[i + 1] = disp[i] + du
        vel[i + 1] = vel[i] + dv
        acc[i + 1] = acc[i] + da
        force[i + 1] = force_c

        # 顺便计算最大绝对加速度、最大速度和最大位移
        if abs(acc[i + 1] + ag[i + 1]) >= Amax:
            Amax = abs(acc[i + 1] + ag[i + 1])
        if abs(vel[i + 1]) >= Vmax:
            Vmax = abs(vel[i + 1])
        if abs(disp[i + 1]) >= Dmax:
            Dmax = abs(disp[i + 1])
            # 返回结构反应
    return acc, vel, disp, force


def plot_timehistory(time_data, y, T, h, cmname):
    # 绘制系统的响应曲线
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(time_data, y, label=cmname + 'T = ' + str(T) + ', h = ' + str(h))
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Duhamel Integral Method')
    plt.legend()
    plt.show()


def plot_disp_force(disp, force, T, h, cmname):
    # 绘制系统的响应曲线
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(disp, force, label=cmname + 'T = ' + str(T) + ', h = ' + str(h))
    plt.xlabel('Disp (m)')
    plt.ylabel('Force (N)')
    plt.title('SDOF_Nonlinear')
    plt.legend()
    plt.show()


def save_to_excel(time, ag, acc, vel, disp, force, scale_factor, T, ksai, cmname):
    # 将数据转换成DataFrame格式
    df = pd.DataFrame(
        {'Time (s)': time, 'Ag (m/s2)': ag, 'Acc (m/s2)': acc, 'Vel (m/s)': vel, 'Disp (m)': disp, 'Force (N)': force})
    # 指定文件名和表名
    file_name = 'SDOF_Nonlinear_' + cmname + '_T=' + str(T) + '_h=' + str(ksai) + '.xlsx'
    sheet_name = 'Sheet1'
    # 保存为xlsx文件
    df.to_excel(file_name, sheet_name=sheet_name, index=False)


def main():
    props = np.zeros(3)
    # 输入地震动路径和文件名
    filename = 'E:\\1\LGP000.AT2'
    # 输入结构参数
    T = 0.5  # 周期（s）
    m = 0.821  # 质量（kg）
    strength = 2.27  # 屈服承载力（N）
    eta = 0.01  # 屈服后刚度系数
    ksai = 0.05  # 阻尼比
    cmname = 'UHYST01'  # 滞回模型编号
    # 设置地震动调幅系数
    scale_factor = 9.8 * 1;

    time, ag, dt, npts = get_peer_data(filename)
    ag = ag * scale_factor
    ag = np.append(ag, 0)
    # 计算刚度和阻尼系数
    k = m / (T / (2 * np.pi)) ** 2
    c = 2 * np.sqrt(m * k) * ksai
    # 生成滞回模型所需的参数向量
    props = k, strength, eta
    # 调用Newmark-beta法求解反应
    acc, vel, disp, force = response(ag, dt, npts, T, m, c, k, props, cmname)
    # 绘制位移反应时程
    plot_timehistory(time, disp, T, ksai, cmname)
    # 绘制力-位移滞回曲线
    plot_disp_force(disp, force, T, ksai, cmname)
    # 将结果保存至Excel文件
    save_to_excel(time, ag, acc, vel, disp, force, scale_factor, T, ksai, cmname)


if __name__ == "__main__":
    main()