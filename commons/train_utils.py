from numpy import real


def check_pseudo_indexes(data):
    pseudo_data = []
    real_data = []
    for i in range(len(data)):
        if data[i]==1:
            real_data.append(i)
        elif data[i]==0:
            pseudo_data.append(i)
    if len(real_data)==0:
        real_data = None
    if len(pseudo_data)==0:
        pseudo_data = None
    return pseudo_data, real_data
