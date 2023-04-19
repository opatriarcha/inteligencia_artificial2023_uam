
notas = [100,50,25,10,5,1]
# notas = [200, 50, 25, 10, 5, 2]

def troco_min(valor, total=0):
	for i in range(len(notas)):
		troco = valor // notas[i]
		valor -= troco * notas[i]
		total += troco
		print('notas: %d [%d]'%(troco, notas[i]))
	print('Total de notas:',total)


troco_min(333)
