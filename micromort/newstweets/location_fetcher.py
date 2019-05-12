import carmen


class Location_fetcher:
    def __init__(self):
        self.resolver = carmen.get_resolver()
        self.resolver.load_locations()
    
    def get_location(self,doc):
        location = self.resolver.resolve_tweet(doc)
        if location is not None:
            return location
        else:
            return None


