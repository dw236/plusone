import util

def crp(L=10, gamma=1.0):
    print "using gamma", gamma
    tables = Tables(gamma)
    for l in range(1, L):
        tables.new_customer()
    return tables.sparse_seats()

class Table(object):
    def __init__(self, customers):
        self.customers = customers
    
    def new_customer(self):
        self.customers += 1     

class Tables(object):
    def __init__(self, gamma=1.0):
        #print "first customer sits at table 0"
        self.gamma = gamma
        self.customers = 1
        self.seats = [0]
        self.tables = [Table(1)]
        
    def new_customer(self):
        self.customers += 1
        #print "new customer has arrived!", self.customers
        probabilities = self.get_probabilities()
        #print "customers are sitting at tables", self.sparse_seats()
        #print "customer will sit with probabilities:", probabilities
        cdf = util.get_cdf(probabilities)
        table = util.sample(cdf)
        #print "customer has chosen to sit at table", table
        if table not in self.seats:
            self.tables.append(Table(1))
        else:
            self.tables[table].new_customer()
        self.seats.append(table)
        
    def get_probabilities(self):
        probabilities = []
        denominator = self.gamma + self.customers - 1
        for table in self.tables:
            numerator = table.customers
            probabilities.append(numerator / denominator)
        probabilities.append(self.gamma / denominator)
        
        return probabilities
    
    def sparse_seats(self):
        return [table.customers for table in self.tables]
    
if __name__ == "__main__":
    print crp()