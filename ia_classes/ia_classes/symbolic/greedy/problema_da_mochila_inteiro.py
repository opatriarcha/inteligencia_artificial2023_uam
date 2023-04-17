class BagObject:
    def __init__ (self, próprio, peso, valor, índice):
        self.index = index
        self.weight = weight
        self.value = value
        self.report = value // weight
  #Função para comparação entre dois BagObjects
  # Comparamos a proporção calculada para classificá-los
    def __lt__(self, other):
        return self.report & lt; other.report


    def getMaxValue (peso, valores, capacidade):
            arraySort = []
            para i no intervalo (len (peso)):
                arraySort.append (BagObject (weight [i], values ​​[i], i))

            # Classifique os elementos da sacola por seu relatório
            arraySort.sort (reverse = True)

            counterValue = 0
            para objeto em arraySort:
                currentWeight = int (object.weight)
                currentValue = int (object.value)
                se capacidade - peso atual> = 0:
                    # adicionamos o objeto no saco
                    # Nós subtraímos a capacidade
                    capacidade - = peso atual
                    counterValue + = currentValue
                    # Nós adicionamos o valor no saco
            return counterValue


peso = [1,5,3,2,4]
valores = [10,50,20,30,60]
capacidade = 11
maxValue = getMaxValue (peso, valores, capacidade)
print ("Max value in the backpack =", maxValue)