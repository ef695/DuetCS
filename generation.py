class ModelConfig(object):
    code_file = sys.argv[1]
    weight_file = 'code_model.h5'
    max_len = 1000
    batch_size = 64
    learning_rate = 0.003


def preprocess_data(ModelConfig):
    files_content = ''
    with open(ModelConfig.code_file, 'r') as f:
        for line in f:
            x = line.strip() + "]"  
            x = x.split(":")[1]

    words = sorted(list(files_content)) 
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    delete_words = []
    for key in counted_words:
        if counted_words[key] <= 2:
            delete_words.append(key)
    for key in delete_words:
        del counted_words[key]
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])   

    words, _ = zip(*wordPairs)
    words += (" ",)
    
    word2idx = dict((c, i) for i, c in enumerate(words))
    idx2word = dict((i, c) for i, c in enumerate(words))
    word2idx_dic = lambda x: word2idx.get(x, len(words) - 1)
    return word2idx_dic, idx2word, words, files_content
	

class LSTMPoetryModel(object):
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.loaded_model = True
        self.config = config

        self.word2idx_dic, self.idx2word, self.words, self.files_content = preprocess_data(self.config)
        
        self.codes = self.files_content.split(']')
        self.codes_num = len(self.codes)
        
        if os.path.exists(self.config.weight_file) and self.loaded_model:
            self.model = load_model(self.config.weight_file)
        else:
            self.train()

    def build_model(self):

        input_tensor = Input(shape=(self.config.max_len, len(self.words)))  
        lstm = LSTM(512, return_sequences=True)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(256)(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(len(self.words), activation='softmax')(dropout)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def sample(self, preds, temperature=1.0):

        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds,1./temperature)
        preds = exp_preds / np.sum(exp_preds)
        prob = np.random.choice(range(len(preds)),1,p=preds)
        return int(prob.squeeze())
    
    def generate_sample_result(self, epoch, logs):
        if epoch % 5 != 0:
            return
        
        for diversity in [0.7, 1.0, 1.3]:
            generate = self.predict_random(temperature=diversity)
            print(generate)
    
    def predict_random(self,temperature = 1):
        if not self.model:
            return
        
        index = random.randint(0, self.codes_num)
        sentence = self.codes[index][: self.config.max_len]
        generate = self.predict_sen(sentence,temperature=temperature)
        return generate
    
    def predict_first(self, char,temperature =1):
        if not self.model:
            return
        
        index = random.randint(0, self.codes_num)
        sentence = self.codes[index][1-self.config.max_len:] + char
        generate = str(char)
        generate += self._preds(sentence,length=23,temperature=temperature)
        return generate
    
    def predict_sen(self, text,temperature =1):
        if not self.model:
            return
        max_len = self.config.max_len
        if len(text)<max_len:
            return

        sentence = text[-max_len:]
        generate = str(sentence)
        generate += self._preds(sentence,length = 24-max_len,temperature=temperature)
        return generate
    
    def predict_hide(self, text,temperature = 1):
        if not self.model:
            return
        if len(text)!=4:
            return
        
        index = random.randint(0, self.codes_num)
        sentence = self.codes[index][1-self.config.max_len:] + text[0]
        generate = str(text[0])
        
        for i in range(5):
            next_char = self._pred(sentence,temperature)           
            sentence = sentence[1:] + next_char
            generate+= next_char
        
        for i in range(3):
            generate += text[i+1]
            sentence = sentence[1:] + text[i+1]
            for i in range(5):
                next_char = self._pred(sentence,temperature)           
                sentence = sentence[1:] + next_char
                generate+= next_char

        return generate
    
    
    def _preds(self,sentence,length = 23,temperature =1):

        sentence = sentence[:self.config.max_len]
        generate = ''
        for i in range(length):
            pred = self._pred(sentence,temperature)
            generate += pred
            sentence = sentence[1:]+pred
        return generate
        
        
    def _pred(self,sentence,temperature =1):
        if len(sentence) < self.config.max_len:
            print('in def _pred,length error ')
            return
        
        sentence = sentence[-self.config.max_len:]
        x_pred = np.zeros((1, self.config.max_len, len(self.words)))
        for t, char in enumerate(sentence):
            x_pred[0, t, self.word2idx_dic(char)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds,temperature=temperature)
        next_char = self.idx2word[next_index]
        
        return next_char

    def data_generator(self):
        i = 0
        while 1:
            x = self.files_content[i: i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]

            if ']' in x or ']' in y:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2idx_dic(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.config.max_len, len(self.words)),
                dtype=np.bool
            )

            for t, char in enumerate(x):
                x_vec[0, t, self.word2idx_dic(char)] = 1.0

            yield x_vec, y_vec
            i += 1

    def train(self):
        number_of_epoch = len(self.files_content)-(self.config.max_len + 1)*self.codes_num
        number_of_epoch /= self.config.batch_size 
        number_of_epoch = int(number_of_epoch / 2.5)

        if not self.model:
            self.build_model()

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )