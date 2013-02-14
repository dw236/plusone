import util

def crp(L=10, gamma=1.0):
    print "using gamma", gamma
    tables = Restaurant(gamma)
    for l in range(1, L):
        tables.new_customer()
    print tables.sparse_seats()
    
    return tables

class Restaurant(object):
    def __init__(self, gamma=1.0):
        print "first customer sits at table 0"
        self.gamma = gamma
        self.customers = 1
        self.seats = [0]
        self.tables = [1]
        
    def new_customer(self):
        self.customers += 1
        
        self.update_probabilities()
        assert(abs(1 - sum(self.probabilities)) < 1e-10)
        
        cdf = util.get_cdf(self.probabilities)
        table = util.sample(cdf)
        
        print "new customer has arrived!", self.customers
        #print "customers are sitting at tables", self.sparse_seats()
        #print "customer will sit with probabilities:", self.probabilities
        print "customer has chosen to sit at table", table
        
        if table not in self.seats:
            self.tables.append(1)
        else:
            self.tables[table] += 1
        self.seats.append(table)
        
    def update_probabilities(self):
        """SHOULD ONLY BE CALLED BY new_customer()"""
        probabilities = []
        denominator = self.gamma + self.customers - 1
        for table in self.tables:
            probabilities.append(table / denominator)
        probabilities.append(self.gamma / denominator)
        
        self.probabilities = probabilities
    
    def sparse_seats(self):
        return self.tables
    
if __name__ == "__main__":
    tables = crp()