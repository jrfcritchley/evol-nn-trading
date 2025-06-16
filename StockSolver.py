import os
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import linecache
import neat


tradeEvents = 10
#tradeEvents is the number of time the network has an option to trade
tradeInterval = 10
#have the multiple of these two equal to ≈50 or something as longer term
#trader strategies limits the pool of stocks that can be picked from with maxlookback
#tradeInterval is the amount of time between trades

maxEpochs = 100

maxLookBack = 125
#maximum amount of time that will be looked back on

folderPath = "C:/Users/Jack Critchley/Desktop/StockSolver/Stocks"

#for encapsulation purposes loadStock loads and converts the stock into a usable format, in most other scenarios
#this would be two functions but then theres a chance of calling the random function again and accidentally operating
#on two completley different stocks as though they were one and neither me or the network will be able to tell easily
#if that happened

def loadStock(folder):
    stockList = os.listdir(folder)
    stockfilePath = os.path.join(folder,random.choice(stockList))
    with open(stockfilePath, "r") as file:
        lineCount = 0
        for line in file:
            lineCount += 1

    firstTrade = random.randint((maxLookBack+2),(lineCount-((tradeEvents*tradeInterval)-1)))
    #the 150 is to ensure that the line gotten can be looked back on by quite a bit (≈7 months) without
    #indexing out of bounds and the lineCount - 51 is to ensure that as the loop progresses it doesnt go out
    
    totalPastTimelines = []
    tradeTimeline = []

    for i in range(tradeEvents):
        line = linecache.getline(stockfilePath,(firstTrade+(tradeInterval*i)))
        yesLine = linecache.getline(stockfilePath,(firstTrade+(tradeInterval*i))-1)
        weekLine = linecache.getline(stockfilePath,(firstTrade+(tradeInterval*i))-5)
        monLine = linecache.getline(stockfilePath,(firstTrade+(tradeInterval*i))-30)
        yearLine = linecache.getline(stockfilePath,(firstTrade+(tradeInterval*i))-maxLookBack)

        #the values for input nodes are based on looking back an arbitrary amount of time as it doesn't matter
        #to the network and these ones seemed right, also the repetition may seem bad but the computational
        #expense is made up for by my mental expense as I don't have to look at bizzare for loops in a couple months
        #when trying to read this
        
        lineSplit = line.strip().split(",")
        yesSplit = yesLine.strip().split(",")
        weekSplit = weekLine.strip().split(",")
        monSplit = monLine.strip().split(",")
        yearSplit = yearLine.strip().split(",")
        pastTimeline = [yesSplit[1],weekSplit[1],monSplit[1],yearSplit[1]]
        totalPastTimelines.append(pastTimeline)
        tradeTimeline.append(lineSplit[1])
        #the 1 here dictates what is used for the stock price: 1 is market open, 2 is market high, 3 is market low and 4 is market close
    return (tradeTimeline,totalPastTimelines)
       
class Stock:  
    def __init__(self,tradeDetails,pastTradeDetails):
        self.tradeDetails = tradeDetails
        self.pastTradeDetails = pastTradeDetails
        self.balance = 1000
        self.quantity = 0

    def invest(self):
            self.quantity += (self.balance / self.current)
            self.balance = 0

    def halfInvest(self):
            self.quantity += ((self.balance / 2) / self.current)  
            self.balance = self.balance/2
        
    def sell(self):
        self.balance += (self.quantity * self.current)
        self.quantity = 0
        
    def halfSell(self):
        self.balance += ((self.quantity / 2) * self.current)
        self.quantity = self.quantity/2
    
    def portfolioEval(self):
        #return ((self.quantity * self.current) + self.balance)
        return(self.balance)
    
    def trainNN(self, genome, config):

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.genome = genome

        #the loop below is to get the new trades for the network at each point it wants to trade
        for i in range(tradeEvents):
            self.current = float(self.tradeDetails[i])
            self.yesterday = float(self.pastTradeDetails[i][0])
            self.week = float(self.pastTradeDetails[i][1])
            self.month = float(self.pastTradeDetails[i][2])
            self.year = float(self.pastTradeDetails[i][3])
            

            netOutput = net.activate((self.current,self.yesterday,self.week,self.month,self.year))
            decision = netOutput.index(max(netOutput))
            #The decision is ouput as an array of the nodes and you get the max value of the nodes to make the decision
            
            if decision == 0:
                 genome.fitness -= 0.01
                 #this is to symbolize making no choice and I want to discourage it as thats a relativley poor investing strategy
                 
            elif decision == 1:
                if self.balance == 0:
                    genome.fitness -= 1
                #this is to symbolize trying to invest with no money and I want to discourage it as thats a relativley poor investing strategy
                else:
                    self.invest()
                    
            elif decision == 2:
                if self.balance == 0:
                    genome.fitness -= 1
                else:
                    self.halfInvest()
                
            elif decision == 3:
                if self.quantity == 0:
                    genome.fitness -= 1
                #this is to symbolize trying to sell assets with no assets and I want to discourage it as thats a relativley poor investing strategy
                else:
                    self.sell()
                    
            else:
                if self.quantity == 0:
                    genome.fitness -= 1
                else:
                    self.halfSell()
            
        genome.fitness = self.portfolioEval()
             
def eval_genomes(genomes,config):
    
    for i, (genome_id, genome) in enumerate(genomes):
        #if i == len(genomes) - 1:
          #  break
            #This is to stop index out of bounds errors
          
        genome.fitness = 0
        #this is to stop unassigned fitness comparison
        allTradeDetails = loadStock(folderPath)
        
        trader = Stock(allTradeDetails[0],allTradeDetails[1])
        trader.trainNN(genome, config)



#if you have to stop the program manually you can go from a checkpoint by uncommenting the first line
#in the function below (and swapping the placeholder checkpoint) and commenting in the second 
def run_neat(config):
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, maxEpochs)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

##THIS IS ALL CODE FOR THE GRAPH
#-----------------------------------------------------------------------------------------------------



    sns.set_style("whitegrid")


    fitness_scores = stats.get_fitness_mean()
    generations = np.arange(1, len(fitness_scores) + 1)


    plt.figure(figsize=(10, 6))


    sns.lineplot(x=generations, y=fitness_scores, marker='o', linewidth=2.5, markersize=10, color='teal')


    plt.title('Average Final Balance of Populations Over Generations', fontsize=18, fontweight='bold', color='teal')
    plt.xlabel('Generation', fontsize=14, fontweight='bold', color='teal')
    plt.ylabel('Average Final Balance of Population', fontsize=14, fontweight='bold', color='teal')


    ticks = np.arange(1, max(generations)+1, 50)
    plt.xticks(ticks, fontsize=12, fontweight='bold', color='teal')
    plt.yticks(fontsize=12, fontweight='bold', color='teal')


    max_fitness = max(fitness_scores)
    max_gen = generations[np.argmax(fitness_scores)]
    plt.annotate(f'Peak ({max_gen}, {max_fitness:.2f})', xy=(max_gen, max_fitness), xytext=(max_gen+0.5, max_fitness+0.5),
                 arrowprops=dict(facecolor='teal', shrink=0.05), fontsize=12, color='teal')

    plt.tight_layout()
    plt.show()

#the bit that runs it

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    run_neat(config)
