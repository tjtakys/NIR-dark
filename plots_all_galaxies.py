#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:09:07 2025

@author: zlemoult
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def donnees_ALPINE():
    names = np.array([
    "CANDELS_GOODSS_32", "DEIMOS_COSMOS_396844", "DEIMOS_COSMOS_422677", "DEIMOS_COSMOS_539609", "DEIMOS_COSMOS_683613",
    "DEIMOS_COSMOS_818760", "DEIMOS_COSMOS_848185", "DEIMOS_COSMOS_873756", "DEIMOS_COSMOS_881725", "vuds_cosmos_5100969402",
    "vuds_cosmos_5101209780", "vuds_cosmos_5101218326", "vuds_cosmos_5180966608", "vuds_efdcs_530029038",
    "DEIMOS_COSMOS_434239", "DEIMOS_COSMOS_454608", "DEIMOS_COSMOS_627939", "DEIMOS_COSMOS_630594", "DEIMOS_COSMOS_845652",
    "vuds_cosmos_5100541407", "vuds_cosmos_5100559223", "vuds_cosmos_510786441", "vuds_cosmos_5110377875"])
    
    z = np.array([
    4.41, 4.54, 4.44, 5.18, 5.54, 4.56, 5.29, 4.55, 4.58, 4.58, 4.57, 4.57,
    4.53, 4.43, 4.49, 4.58, 4.53, 4.44, 5.31, 4.56, 4.56, 4.46, 4.55])
    
    irx = np.array([
    (0.57, 0.14), (0.57, 0.14), (0.63, 0.11), (0.07, 0.22), (0.51, 0.17), (0.81, 0.14),
    (0.32, 0.09), (1.36, 0.22), (0.66, 0.13), (0.54, 0.19), (0.17, 0.26), (0.51, 0.14),
    (0.76, 0.13), (0.00, 0.16), (0.61, 0.08), (0.33, 0.08), (0.48, 0.08), (0.55, 0.09),
    (-0.20, 0.10), (1.04, 0.09), (0.40, 0.18), (-0.10, 0.10), (0.65, 0.05)])
    
    beta = np.array([
    (-1.20, 0.08), (-1.44, 0.24), (-1.29, 0.20), (-2.42, 0.11), (-1.90, 0.22), (-0.74, 0.17),
    (-1.20, 0.17), (-1.31, 0.31), (-1.20, 0.22), (-1.85, 0.25), (-2.16, 0.15), (-0.98, 0.18),
    (-0.80, 0.24), (-1.92, 0.07), (-1.25, 0.26), (-1.45, 0.20), (-1.46, 0.24), (-1.64, 0.25),
    (-1.73, 0.10), (-1.94, 0.26), (-2.00, 0.22), (-2.00, 0.10), (-1.38, 0.18)])
    """
    delta = np.array([
    (-0.84, 0.31), (-0.62, 0.42), (-0.67, 0.32), None, (0.02, 0.33), (-0.72, 0.26),
    (-1.44, 0.36), None, (-0.67, 0.30), (-0.12, 0.40), None, (-0.85, 0.30),
    (-0.68, 0.29), (-0.64, 0.45), (-0.71, 0.33), (-1.23, 0.43), (-0.75, 0.44), (-0.41, 0.36),
    (-1.84, 0.17), None, (-0.08, 0.45), (-1.19, 0.45), (-0.55, 0.23)])
    
    afuv = np.array([
    (1.46, 0.24), (1.42, 0.23), (1.54, 0.20), (0.67, 0.21), (1.28, 0.26), (1.94, 0.26),
    (1.05, 0.14), (3.03, 0.49), (1.60, 0.23), (1.36, 0.30), (0.81, 0.28), (1.34, 0.24),
    (1.79, 0.26), (0.62, 0.17), (1.49, 0.16), (1.02, 0.13), (1.25, 0.13), (1.37, 0.15),
    (0.43, 0.09), (2.25, 0.19), (1.11, 0.26), (0.48, 0.09), (1.56, 0.09)])
    """
    Av = np.array([
    (0.24, 0.10), (0.32, 0.18), (0.31, 0.13), (0.39, 0.20), (0.58, 0.23), (0.35, 0.13),
    (0.08, 0.05), (1.40, 0.47), (0.31, 0.13), (0.55, 0.27), (0.40, 0.23), (0.20, 0.09),
    (0.33, 0.13), (0.14, 0.09), (0.28, 0.12), (0.11, 0.07), (0.24, 0.14), (0.38, 0.17),
    (0.02, 0.01), (1.25, 0.26), (0.49, 0.27), (0.05, 0.04), (0.34, 0.11)])
    
    SFR = np.array([
    (50.86, 15.89), (76.88, 18.75), (91.97, 18.88), (66.52, 13.38), (54.64, 14.19), (159.53, 53.88),
    (120.35, 30.07), (142.62, 66.46), (87.12, 23.58), (59.00, 21.63), (54.90, 18.01), (82.33, 20.71),
    (70.99, 17.94), (32.50, 6.47), (103.13, 15.75), (62.19, 8.09), (65.79, 5.93), (57.27, 5.52),
    (72.76, 7.19), (99.83, 11.58), (34.92, 8.01), (61.60, 6.62), (161.06, 9.09)])
    
    SM = np.array([
    (7.52e9, 1.83e9), (7.84e9, 2.27e9), (8.43e9, 1.92e9), (5.41e9, 1.46e9), (1.95e10, 7.7e9),
    (5.07e10, 9.63e9), (2.42e10, 5.32e9), (4.26e10, 1.88e10), (1.04e10, 2.5e9), (1.24e10, 4.58e9),
    (1.84e10, 4.73e9), (1.22e11, 1.8e10), (7.17e10, 1.32e10), (1.51e10, 2.35e9), (2.74e10, 7.85e9),
    (6.04e9, 1.55e9), (1.03e10, 3.42e9), (6.28e9, 2.09e9), (5.08e10, 7.7e9), (3.30e10, 1.56e10),
    (9.51e9, 3.16e9), (1.18e10, 2.18e9), (1.41e10, 3.62e9)])
    
    if not len(SM) == len(SFR) == len(Av):
        print("Les donnéees d'ALPINE ne sont pas à la bonne taille")
    
    return [z, irx, beta, Av, SFR, SM, names]

def extract_mstar_av(file_path = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//Smail+21_Mstar_Av.csv'):
    """
    Extrait les colonnes Mstar et Av du fichier Smail+21_Mstar_Av.csv.

    Paramètres :
        file_path (str) : chemin du fichier CSV.

    Retourne :
        tuple : deux listes (Mstar, Av)
    """
    # Lire le fichier en ignorant la première ligne
    df = pd.read_csv(file_path, delim_whitespace = True, skiprows=1, names=["Mstar", "Av", "Flag"])
    
    # Extraire les colonnes
    mstar_list = np.log10(df["Mstar"].tolist())
    av_list = df["Av"].tolist()
    
    return mstar_list, av_list

def plot_SFR_M(SFR, SFR_err, M, M_err, zspecs, galaxy_names):
    marq = ['^', '*', 'd']
    SFR_alpine, SFR_alpine_err = np.log10(donnees_ALPINE()[4][:, 0]), donnees_ALPINE()[4][:, 1] / (donnees_ALPINE()[4][:, 0] * np.log(10))
    M_alpine, M_alpine_err = np.log10(donnees_ALPINE()[5][:, 0]), donnees_ALPINE()[5][:, 1] / (donnees_ALPINE()[5][:, 0] * np.log(10))
    log_SFR_err = SFR_err / (SFR * np.log(10))
    log_SFR = np.log10(SFR)
    fig, ax = plt.subplots(figsize = (15, 10))

    M_min, M_max = min(min(M), min(M_alpine)), max(max(M), max(M_alpine))
    # SFR_min, SFR_max = min(min(SFR), min(SFR_alpine)), max(max(SFR), max(SFR_alpine))
    X = np.linspace(M_min, M_max, 5000)
    m0 = 0.5
    a0 = 1.5
    a1 = 0.3
    m1 = 0.36
    a2 = 2.5
    for j in [3, 4, 5]:
        r = np.log10(1 + j)
        y = [m - 9 - m0 + a0*r - a1 * max(0, m - 9 - m1 - a2 * r)**2 for m in X]
        ax.plot(X, y, 'r', label = f"MS z = {j}")
    
    # ax.scatter(M_alpine, SFR_alpine, color = 'r', marker = 's', label = 'ALPINE')
    ax.errorbar(M_alpine, SFR_alpine, xerr = M_alpine_err, yerr = SFR_alpine_err, fmt = 'gs', capsize = 3, alpha = 0.6, linestyle = 'none', label = 'ALPINE')
    ax.set_ylabel(r'SFR [$ \log_{10} \left( M_\odot \mathrm{yr}^{-1} \right)$]', size = 20)
    ax.set_xlabel(r'Stellar Mass [$\log_{10} \left( \frac{M_*}{M_\odot} \right)$]', size = 20)
    
    for i, gal_name in enumerate(galaxy_names):
        ax.errorbar(M[i], log_SFR[i], xerr = M_err[i], yerr = log_SFR_err[i], color = 'blue' , marker = marq[i], capsize = 3, alpha = 1, linestyle = 'none', label = gal_name, markersize = 12)
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    ax.legend(fontsize=12)
    fig.canvas.manager.set_window_title("SFR vs M* for all galaxies")
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "SFR_vs_M*_all_galaxies.png"), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def plot_Av_M(Av, Av_err, M, M_err, zspecs, galaxy_names):
    marq = ['^', '*', 'd']
    Av_alpine, Av_alpine_err = donnees_ALPINE()[3][:, 0], donnees_ALPINE()[3][:, 1]
    M_alpine, M_alpine_err = np.log10(donnees_ALPINE()[5][:, 0]), donnees_ALPINE()[5][:, 1] / (donnees_ALPINE()[5][:, 0] * np.log(10))
    M_Smail, Av_Smail = extract_mstar_av(file_path = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//Smail+21_Mstar_Av.csv')
    fig, ax = plt.subplots(figsize = (15, 10))
    
    # ax.scatter(M_alpine, Av_alpine, color = 'r', marker = 's', label = 'ALPINE')
    ax.errorbar(M_alpine, Av_alpine, xerr = M_alpine_err, yerr = Av_alpine_err, fmt = 'gs', capsize = 3, alpha = 0.6, linestyle = 'none', label = 'ALPINE')
    ax.scatter(M_Smail, Av_Smail, color = 'r', marker = '^', label = 'Smail and al.', alpha = 0.6)
    ax.set_ylabel(r'Av [mag]', size = 20)
    ax.set_xlabel(r'Stellar Mass [$\log_{10} \left( \frac{M_*}{M_\odot} \right)$]', size = 20)
    for i, gal_name in enumerate(galaxy_names):
        ax.errorbar(M[i], Av[i], xerr = M_err[i], yerr = Av_err[i], color = 'blue' , marker = marq[i], capsize = 3, alpha = 1, linestyle = 'none', label = gal_name, markersize = 12)

    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    ax.legend(fontsize=12)
    fig.canvas.manager.set_window_title("Av vs M* for all galaxies")
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "Av_vs_M*_all_galaxies.png"), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def plot_IRX_β(IRX, IRX_err, β, β_err, zspecs, galaxy_names):
    marq = ['^', '*', 'd']
    IRX_alpine, IRX_alpine_err = np.log10(donnees_ALPINE()[1][:, 0]), donnees_ALPINE()[1][:, 1] / (donnees_ALPINE()[1][:, 0] * np.log(10))
    β_alpine, β_alpine_err = donnees_ALPINE()[2][:, 0], donnees_ALPINE()[2][:, 1]
    log_IRX_err = IRX_err / (IRX * np.log(10))
    log_IRX = np.log10(IRX)
    fig, ax = plt.subplots(figsize = (15, 10))
    
    β_min, β_max = min(min(β), min(β_alpine)), max(max(β), max(β_alpine))
    # IRX_min, IRX_max = min(min(IRX), min(IRX_alpine)), max(max(IRX), max(IRX_alpine))
    X = np.linspace(β_min, β_max, 5000)
    y_starbust = [np.log10(1.67 * (10**(0.4 * (2.13 * m + 5.57)) - 1)) for m in X]
    y_SMC = [np.log10(1.79 * (10**(0.4 * (1.07 * m + 2.79)) - 1)) for m in X]

    # ax.scatter(M_alpine, IRX_alpine, color = 'r', marker = 's', label = 'ALPINE')
    ax.errorbar(β_alpine, IRX_alpine, xerr = β_alpine_err, yerr = abs(IRX_alpine_err), fmt = 'ms', capsize = 3, alpha = 0.6, linestyle = 'none', label = 'ALPINE')
    ax.set_ylabel(r'$\log_{10} \left( IRX \right)$', size = 20)
    ax.set_xlabel(r'β', size = 20)
    for i, gal_name in enumerate(galaxy_names):
        ax.errorbar(β[i], log_IRX[i], xerr = β_err[i], yerr = log_IRX_err[i], color = 'blue' , marker = marq[i], capsize = 3, alpha = 1, linestyle = 'none', label = gal_name, markersize = 12)
    ax.plot(X, y_starbust, 'r', label = 'Starbust')
    ax.plot(X, y_SMC, 'black', label = 'SMC')
    
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    ax.legend(fontsize=12)
    fig.canvas.manager.set_window_title("log10(IRX) vs β for all galaxies ")
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "log10(IRX)_vs_β_all_galaxies.png"), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def plot_IRX_β2(IRX, IRX_err, β, β_err, zspecs, galaxy_names):
    marq = ['^', '*', 'd']
    IRX_alpine, IRX_alpine_err = np.log10(donnees_ALPINE()[1][:, 0]), donnees_ALPINE()[1][:, 1] / (donnees_ALPINE()[1][:, 0] * np.log(10))
    β_alpine, β_alpine_err = donnees_ALPINE()[2][:, 0], donnees_ALPINE()[2][:, 1]
    log_IRX_err = IRX_err / (IRX * np.log(10))
    log_IRX = np.log10(IRX)
    fig, ax = plt.subplots(figsize = (15, 10))
    
    β_min, β_max = min(min(β), min(β_alpine)), max(max(β), max(β_alpine))
    # IRX_min, IRX_max = min(min(IRX), min(IRX_alpine)), max(max(IRX), max(IRX_alpine))
    X = np.linspace(β_min, β_max, 5000)
    y_starbust = [np.log10(1.67 * (10**(0.4 * (2.13 * m + 5.57)) - 1)) for m in X]
    y_SMC = [np.log10(1.79 * (10**(0.4 * (1.07 * m + 2.79)) - 1)) for m in X]

    # ax.scatter(M_alpine, IRX_alpine, color = 'r', marker = 's', label = 'ALPINE')
    ax.errorbar(β_alpine, IRX_alpine, xerr = β_alpine_err, yerr = abs(IRX_alpine_err), fmt = 'ms', capsize = 3, alpha = 0.6, linestyle = 'none', label = 'ALPINE')
    ax.set_ylabel(r'$\log_{10} \left( IRX \right)$', size = 20)
    ax.set_xlabel(r'β', size = 20)
    for i, gal_name in enumerate(galaxy_names):
        ax.errorbar(β[i], log_IRX[i], xerr = β_err[i], yerr = log_IRX_err[i], color = 'blue' ,marker = marq[i], capsize = 3, alpha = 1, linestyle = 'none', label = gal_name, markersize = 12)
    ax.plot(X, y_starbust, 'r', label = 'Starbust')
    ax.plot(X, y_SMC, 'black', label = 'SMC')
    
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    ax.set_xlim(-3, 0)   # Pour les abscisses (x)
    ax.set_ylim(-1, 3)   # Pour les ordonnées (y)
    ax.legend(fontsize=12)
    fig.canvas.manager.set_window_title("log10(IRX) vs β for all galaxies zoom")
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "log10(IRX)_vs_β_all_galaxies_zoom.png"), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def get_maps_pxls(name_file):
    Rv = 3.1 # from cigale
    # M_sun = 1.9885e30  # [kg]
    L_sun = 3.828e26 # [W]
    # Replace this path with the path to your file (csv or tsv)
    df = pd.read_csv(name_file, sep=r'\s+|,', engine='python')
    # Extracting columns
    sfr       = df['bayes.sfh.sfr'].to_numpy()
    sfr10M   = df['bayes.sfh.sfr10Myrs'].to_numpy()
    sfr100M   = df['bayes.sfh.sfr100Myrs'].to_numpy()
    sm = np.log10(df['bayes.stellar.m_star'].to_numpy())
    e_bv      = df['bayes.attenuation.E_BV_lines'].to_numpy() * Rv
    chi2_red  = df['best.reduced_chi_square'].to_numpy()
    dust_lum = np.log10(df['bayes.dust.luminosity'].to_numpy() / L_sun)
    age_burst = df['bayes.sfh.age_burst'].to_numpy()
    f_burst = df['bayes.sfh.f_burst'].to_numpy()
    tau_main_sfh = df['bayes.sfh.tau_main'].to_numpy()
    UV_slope = df['bayes.attenuation.powerlaw_slope'].to_numpy()
    beta      = df['bayes.param.beta_calz94'].to_numpy()
    irx       = df['bayes.param.IRX'].to_numpy()
    
    sfr_err       = df['bayes.sfh.sfr_err'].to_numpy()
    sfr10M_err   = df['bayes.sfh.sfr10Myrs_err'].to_numpy()
    sfr100M_err   = df['bayes.sfh.sfr100Myrs_err'].to_numpy()
    sm_err = df['bayes.stellar.m_star_err'].to_numpy() / (df['bayes.stellar.m_star'].to_numpy() * np.log(10))
    e_bv_err      = df['bayes.attenuation.E_BV_lines_err'].to_numpy() * Rv
    dust_lum_err = df['bayes.dust.luminosity_err'].to_numpy() / (df['bayes.dust.luminosity'].to_numpy() / np.log(10))
    age_burst_err = df['bayes.sfh.age_burst_err'].to_numpy()
    f_burst_err = df['bayes.sfh.f_burst_err'].to_numpy()
    tau_main_sfh_err = df['bayes.sfh.tau_main_err'].to_numpy()
    UV_slope_err = df['bayes.attenuation.powerlaw_slope_err'].to_numpy()
    beta_err       = df['bayes.param.beta_calz94_err'].to_numpy()
    irx_err       = df['bayes.param.IRX_err'].to_numpy()

    # Store in a list in the desired order
    results = [sfr, sfr10M, sfr100M, sm, e_bv, chi2_red, dust_lum,
               age_burst, f_burst, tau_main_sfh, UV_slope, irx, beta]
    err_results = [sfr_err, sfr10M_err, sfr100M_err, sm_err, e_bv_err, np.zeros(1), dust_lum_err,
                   age_burst_err, f_burst_err, tau_main_sfh_err, UV_slope_err, irx_err, beta_err]
    names = [r'SFR [$M_\odot \mathrm{yr}^{-1}$]',
    r'SFR_10M [$M_\odot \mathrm{yr}^{-1}$]',
    r'SFR_100M [$M_\odot \mathrm{yr}^{-1}$]',
    r'Stellar Mass [$\log_{10} \left( \frac{M_*}{M_\odot} \right)$]', 
    r'Av [mag]',
    r'red_chi²',
    r'Dust luminosity [$\log_{10} \left( \frac{L}{L_\odot} \right)$]',
    r'Age burst [Myr]',
    r'f burst [0 : 1]',
    r'Tau_main [Myr]',
    r'UV_slope',
    r'IRX',
    r'β']
    
    """
    # Exemple d’utilisation
    for i, arr in enumerate(results):
        print(f'Tableau {i+1}, premiers 5 éléments :', arr[:5])
    """
    return results, err_results, names

def main():
    """Main function to run CIGALE and process images' results."""
    result_file = os.path.join("out", "results.txt")
    
    Galaxy_names = ['R0600-ID67', 'A0102-ID224', 'M0417-ID46']
    Galaxies_zspec = [4.80, 4.33, 3.65]
    direcs = ['//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test//',
              '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//A0102-ID224//JWST', 
              '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//M0417-ID46//JWST']
    maps_list = []
    maps_err_list = []
    
    for i, direc in enumerate(direcs):
        os.chdir(direc) # Change to working directory
                
        # Step 0.5 : Get the maps of the spatially resolved SED Analysis
        maps, maps_err, name_maps = get_maps_pxls(name_file = result_file)
        values = [v.item() for v in maps]
        errors = [e.item() for e in maps_err]
        maps_list.append(values)
        maps_err_list.append(errors)
        
    values_list = np.array(maps_list)
    values_err_list = np.array(maps_err_list)
    
    os.chdir("//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//") # Change to working directory
    
    # === Print SFR vs Stellar Mass plot and other ===
    SFR, SFR_err = values_list[:, 0], values_err_list[:, 0]
    M, M_err = values_list[:, 3], values_err_list[:, 3]
    Av, Av_err = values_list[:, 4], values_err_list[:, 4]
    IRX, IRX_err = values_list[:, -2], values_err_list[:, -2]
    β, β_err = values_list[:, -1], values_err_list[:, -1]
    plot_SFR_M(SFR, SFR_err, M, M_err, Galaxies_zspec, Galaxy_names)
    plot_Av_M(Av, Av_err, M, M_err, Galaxies_zspec, Galaxy_names)
    plot_IRX_β(IRX, IRX_err, β, β_err, Galaxies_zspec, Galaxy_names)
    plot_IRX_β2(IRX, IRX_err, β, β_err, Galaxies_zspec, Galaxy_names)
    

if __name__ == '__main__':
    main()