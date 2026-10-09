#!/usr/bin/env python3
"""Generate the README chart directly from the saved simulation output."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent.parent

def main():
    sim = json.loads((ROOT / 'data/simulation_results.json').read_text())
    entries = sim['champion_probabilities']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})
    fig, ax = plt.subplots(figsize=(12,8),facecolor='#101820')
    ax.set_facecolor('#101820')
    values = [e['probability']*100 for e in entries]
    names = [e['team'] for e in entries]
    colors = ['#f3c969' if i<3 else '#68cbb5' if value else '#53616f' for i,value in enumerate(values)]
    ax.barh(names,values,color=colors,height=.66)
    ax.invert_yaxis()
    maximum=max(values)
    for i,value in enumerate(values):
        label=f'{value:.2f}%' if value else '0% · locked elimination'
        ax.text(value+maximum*.022,i,label,color='#e7edf2',va='center',fontsize=10)
    ax.set_xlim(0,maximum*1.38)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
    ax.tick_params(axis='both',colors='#b7c3cf',length=0,pad=9)
    ax.xaxis.grid(True,color='#3b4855',alpha=.5,linewidth=.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel('Share of simulated tournaments won',color='#b7c3cf',labelpad=13)
    fig.text(.06,.947,'WORLD CUP 2026',color='#68cbb5',fontsize=12,weight='bold')
    fig.text(.06,.899,'Champion probabilities',color='#f7fafc',fontsize=26,weight='bold')
    fig.text(.06,.855,f"July 5 snapshot · supplied R16 bracket · {sim['runs']:,} runs · seed {sim['seed']}",color='#b7c3cf',fontsize=11)
    fig.text(.06,.037,'Heuristic simulation estimates. Two supplied July 4 results are locked. Accuracy has not been measured.',color='#98a8b7',fontsize=9)
    fig.subplots_adjust(left=.22,right=.94,top=.8,bottom=.13)
    output=ROOT/'docs/champion_probabilities.png'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output,dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Generated {output.name}')

if __name__ == '__main__':
    main()
