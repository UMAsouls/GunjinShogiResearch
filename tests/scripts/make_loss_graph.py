
import matplotlib.pyplot as plt
import japanize_matplotlib

import gc

LOSS_DIR = "model_loss"
METHOD_NAME = "deepnash_mp"
NAME = "v6"

def plot_graph(path:str, losses, policies, values):
    fig, ax = plt.subplots(figsize=(12,7))
    fig.suptitle("loss")
        
    ax.plot(losses, label="合計loss")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(f"{path}/all_loss.png", format="png")
    plt.cla()
        
    ax.plot(policies, label="p_log × Qの平均")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(f"{path}/logit_q.png", format="png")
    plt.cla()
        
    ax.plot(values, label = "v_loss")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(f"{path}/v_loss.png", format="png")
    plt.cla()
        
    plt.clf()
    plt.close("all")
    gc.collect()

def make_loss_graph(path:str):

    with open(f"{path}/loss.csv", "r") as f:
        lines = f.readlines()
        f.close()

    losses = []
    policies = []
    values = []

    for line in lines[1:]:
        datas = line.split(",")
        losses.append(float(datas[0]))
        policies.append(float(datas[1]))
        values.append(float(datas[2]))

    plot_graph(path, losses, policies, values)

if __name__ == "__main__":
    make_loss_graph(f"{LOSS_DIR}/{METHOD_NAME}/{NAME}")