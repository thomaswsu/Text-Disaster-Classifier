class Tweet(object):
    """ Created to represent a Tweet in a maintainable way 
    NOT NEEDED CURRENTLY
    """

    def __init__(self, statusid: int, ttext: str):
        self.Statusid = statusid
        self.Ttext = ttext
        
        self.hashTags = []
        self.urls = []
        for word in self.Ttext.split():
            if(word[0] == '#'):
                self.hashTags.append(word)
            elif(word[0:3] == "www" or word[0:4] == "http" or word[0:5] == "https" or word[0:4] == "Gÿæn"):
                self.urls.append(word)

    
    def __repr__(self):
        return("\n Status ID: {}\n Tweet: {} \n".format(self.Statusid, self.Ttext))


""" Tweets = {} # Make a dictionary of tweets where key is id 

for i in training_data.index / 5:
    Tweets[training_data["statusid"][i]] = Tweet(training_data["statusid"][i], \
        training_data["ttext"][i], training_data["Disaster"][i])
"""
