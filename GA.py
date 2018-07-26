import pandas as pd
import numpy as np 
import struct
from decimal import *
import random








def SolveEquation(x):
    Constant = [1, -7, 12.25]
    Variable = [x*x, x, 1]
    return 1/(np.dot(Constant, Variable)+0.001)

def doparts(n):
    a = Decimal(n)
    a_split = (int(a // 1), a % 1)
    return (a_split[0], a_split[1])


def dec2binary(n):
	i, f = doparts(n)
	b = ''
	while (i):
		if (i % 2 == 0):
			b += '0'
		else:
			b += '1'
		i = i // 2

	b = ''.join(reversed(b))
	fp = ''
	f = f * 2
	for n in range(4):
		p, q = doparts(f)
		fp += str(p)
		f = q * 2
		bi = (float(str(b) + '.' + str(fp)))
	return ('{:08.4f}'.format(bi))

def bin2dec(n):
    a = Decimal(n)
    a_split = (int(a // 1), a % 1)
    i = genetic_code[str(a_split[0])]
    f = genetic_code[str('{:05.4f}'.format(a_split[1]).split('.')[1])]
    r = float(i) + float(f)
    return r

genetic_code = {
    '0': '0',
    '1': '1',
    '10': '2',
    '11': '3',
    '100': '4',
    '101': '5',
    '110': '6',
    '111': '7',
    '0000': '0.0',
    '0001': '0.5',
    '0010': '0.25',
    '0011': '0.75',
    '0100': '0.125',
    '0101': '0.625',
    '0110': '0.375',
    '0111': '0.875',
    '1000': '0.062',
    '1001': '0.562',
    '1010': '0.312',
    '1011': '0.812',
    '1100': '0.187',
    '1101': '0.687',
    '1110': '0.437',
    '1111': '0.937',

}

def dochoies(p, q):
    choices = {chromosome: fitness for chromosome, fitness in zip(p, q)}
    return choices

def weighted_random_choice(choices,s):
    pick = random.uniform(0, s)
    current = 0
    w=[]
    for key, value in choices.items():
        current += value
        w=key
        if current >= pick:
            w=key
            break
    return w

def mutate(ch,pm):
    # ch=str(ch)
    ch1=ch.split('.')[0]
    ch2=ch.split('.')[1]
    ch=ch1+ch2
    mutatedCh = []
    for i in ch:
        if random.random() < pm:
            if i == 1:
                mutatedCh.append('0')
            else:
                mutatedCh.append('1')
        else:
            mutatedCh.append(str(i))
    # assert mutatedCh != ch
    mutatedCh1=''.join(mutatedCh[:3])
    mutatedCh2=''.join(mutatedCh[3:])
    mutatedCh=mutatedCh1+'.'+mutatedCh2
    return mutatedCh


def crossover(ch1, ch2):
    r = random.randint(1, 7)
    return ch1[:r] + ch2[r:], ch2[:r] + ch1[r:]



def generatenewpopulation(newparents,bin,pm):
    newpop = np.array(newparents.value_counts().index)
    np.array(newparents.value_counts()[newparents.value_counts()>1].index)
    newpop=newpop.reshape((len(newpop),1))
    n = np.array(newparents.value_counts()[newparents.value_counts()>1].index)
    n = n.reshape(len(n),1)
    newpop = np.concatenate((newpop,n))
    _ = len(newparents)-len(newpop)

    m=[]
    for i in range(_):
        ch1, ch2 = crossover(dec2binary(newpop.flatten('C')[0]), dec2binary(n.flatten('C')[0]))
        ch1 = bin2dec(ch1)
        ch2 = mutate(ch2,pm)
        ch2 = bin2dec(ch2)
        newpop = np.append(newpop, [ch2,ch1])
    newpop = np.append(newpop, m[:_])

    return(newpop.flatten('C')[:8])



def main():
	solution = [0.500, 1.875, 2.125, 4.875, 5.500, 6.875, 2.500, 3.125]
	pc = 1
	pm = 0.125
	e = 0.001
	itr = 100
	
	df = pd.DataFrame(data=None,columns=['ValueOfX', 'Fitness', 'RouletWheelPopulation','NewPopulation'], index=np.arange(0,len(solution),1))
	
	for i in range(itr):
		df['ValueOfX'] = solution
		df['Fitness'] = df['ValueOfX'].apply(SolveEquation)
		df.sort_values(by=['Fitness'], ascending=True,inplace=True)
		sumoffitness = df['Fitness'].sum()
		fitness_ = df['Fitness']/df['Fitness'].sum()
		InitialPopulation = df['ValueOfX'].apply(dec2binary)
		cos = dochoies(df['ValueOfX'], df['Fitness'])
		df['RouletWheelPopulation'] = df.apply(lambda x:weighted_random_choice(cos, df['Fitness'].sum()), axis=1)
		df['NewPopulation'] = generatenewpopulation(df['RouletWheelPopulation'],InitialPopulation,pm)
		solution = df['NewPopulation']

		# print(df)

	print("Chossen Solution is : {}".format(df['ValueOfX'][df['Fitness'].idxmax()]))


	


if __name__ == "__main__":
    main()
