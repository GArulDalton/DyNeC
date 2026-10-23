import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','Arial','DejaVu Sans'],'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'savefig.dpi':600,'ps.fonttype':42,'pdf.fonttype':42})
def save(fig,name):
    fig.savefig(name+'.png',dpi=600,bbox_inches='tight'); fig.savefig(name+'.eps',bbox_inches='tight')
