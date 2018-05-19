#!/usr/bin/env python
# -*- coding: utf-8 -*- 
training_data = []
training_data.append({"class":"sad","sentence": "I'm feeling down about work lately."})
training_data.append({"class":"sad","sentence": "Tom's upset about his boss. He's too hard on him!"})
training_data.append({"class":"sad","sentence": "I'm sad about the situation at work."})
training_data.append({"class":"sad","sentence": "Peter is out of sorts today. Ask him tomorrow."})
training_data.append({"class":"sad","sentence": "the staff does not feel well about the changes at work."})
training_data.append({"class":"sad","sentence": "Jack is feeling blue about his relationship with his girlfriend."})
training_data.append({"class":"sad","sentence": "Kelly is in the dumps about her horrible job."})
training_data.append({"class":"sad","sentence": "Keith feels down in the mouth about his relationship."})
training_data.append({"class":"sad","sentence": "i went through a breakup"})

training_data.append({"class":"angery","sentence": "Mind your own business!"})
training_data.append({"class":"angery","sentence": "It’s none of your business!"})
training_data.append({"class":"angery","sentence": "Shame on you!"})
training_data.append({"class":"angery","sentence": "It makes me see red!"})
training_data.append({"class":"angery","sentence": "I have enough of that boy!"})
training_data.append({"class":"angery","sentence": " I won’t have it!"})
training_data.append({"class":"angery","sentence": "Why are so angry with her?"})
training_data.append({"class":"angery","sentence": " Who do you take me for?"})
training_data.append({"class":"angery","sentence": "For God’s sake, leave me alone!"})
training_data.append({"class":"angery","sentence": "I’m fed up with his lies."})
training_data.append({"class":"angery","sentence": " There is no reason why I should stay here."})
training_data.append({"class":"angery","sentence": "What irritates me most is that nobody believes me."})
training_data.append({"class":"angery","sentence": " I won’t tolerate living among them."})

training_data.append({"class":"happpy","sentence": "When I sign the lease on my new apartment, I’m going to jump for joy!"})
training_data.append({"class":"happpy","sentence": "The food at the five-star restaurant is awesome!"})
training_data.append({"class":"happpy","sentence": "When I graduated from college, I was on cloud nine!"})
training_data.append({"class":"happpy","sentence": "The owner of the company is really generous with vacation time."})
training_data.append({"class":"happpy","sentence": " I’m honored to accept this position in the company and will work hard to make the team proud!"})
training_data.append({"class":"happpy","sentence": "After she said yes to his proposal, he was so happy that he was grinning from ear to ear for weeks after."})
training_data.append({"class":"happpy","sentence": "The cake we had for the holiday was homemade, so it was extraordinarily delicious."})
training_data.append({"class":"happpy","sentence": "Once we lit the fire in the furnace, we all sat down with a cup of chocolate and I was a happy camper."})
training_data.append({"class":"happpy","sentence": "The handmade furniture was perfect for our home and fit just as we expected!"})
training_data.append({"class":"happpy","sentence": "I’m happy to see you back on your feet only a week after your surgery!"})
training_data.append({"class":"happpy","sentence": "Her success in the last three years brought her into the role as a director of the department."})
training_data.append({"class":"happpy","sentence": "All the compliments from my last project were like music to my ears."})
training_data.append({"class":"happpy","sentence": "After her last movie, she became famous in the United States and the U.K."})
training_data.append({"class":"happpy","sentence": "Traveling across the world makes me happy as a clam."})
training_data.append({"class":"happpy","sentence": " I admire people who are generous and kind."})
training_data.append({"class":"happpy","sentence": "The team’s jaws dropped after finishing a three-year project."})
training_data.append({"class":"happpy","sentence": "Traveling is so exciting that it motivates me to work hard."})
training_data.append({"class":"happpy","sentence": "My brother becomes friends with everyone he meets because his personality is larger than life!"})
training_data.append({"class":"happpy","sentence": "It’s courageous when people to stand up for what they believe "})
training_data.append({"class":"happpy","sentence": "I got my college acceptance letter this afternoon, it’s the best day ever!"})

training_data.append({"class":"fear","sentence": "It was such a terrifying ordeal. I’m glad that it’s over."})
training_data.append({"class":"fear","sentence": "I watched a horror movie yesterday. Some of the scenes and the sound effects were so frightening that they sent shivers down my spine."})
training_data.append({"class":"fear","sentence": "I can’t watch horror films. They give me goose bumps."})
training_data.append({"class":"fear","sentence": "If something makes the hairs on the back of your neck stand up, they scare you."})
training_data.append({"class":"fear","sentence": "I get scared really easily."})
training_data.append({"class":"fear","sentence": "I’m afraid of the dark."})
training_data.append({"class":"fear","sentence": " can’t watch horror films. They scare me."})
training_data.append({"class":"fear","sentence": "I had a terrifying experience last week."})


print ("%s sentences of training data" % len(training_data))