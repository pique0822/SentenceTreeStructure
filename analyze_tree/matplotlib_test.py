import matplotlib.pyplot as plt

num_rows = 3
num_cols = 3
fig,axes = plt.subplots(num_rows,num_cols)

x_lower = -10
x_upper = 2

y_lower = -1
y_upper = 20
for r in range(num_rows):
    for c in range(num_cols):
        # ax.title(str(r)+' , '+str(c))
        axes[r,c].scatter(range(r+c+1),range(r+c+1))

        axes[r,c].set_xbound(x_lower,x_upper)
        axes[r,c].set_ybound(y_lower,y_upper)

plt.show()
