from transformers import GPT2LMHeadModel
from distances import Distance, CosineSimilarity
from data import Batch, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import math

def get_gpt_model() -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained("gpt2")

def generate_embeds(input : torch.Tensor, model : GPT2LMHeadModel) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns input and output embeddings
    hidden_states = model(input, output_hidden_states = True).hidden_states
    return hidden_states[0], hidden_states[-1]

    
class GPTDistanceRegression():
    def __init__(self, out_dim : int, 
                 distance : Distance, 
                 optimiser_params : dict = {},
                 training_params : dict = {}):
        
        self.invert : bool = not isinstance(distance, CosineSimilarity)

        self.gpt : GPT2LMHeadModel = get_gpt_model()
        self.distance : Distance = distance
        self.linear : nn.Linear = nn.Linear(1, out_dim, bias = True)

        self.loss : nn.MSELoss = nn.MSELoss()
        #TODO ensure that GPT model is frozen

        self.optimiser : optim.SGD = optim.SGD(list(self.distance.parameters()) + 
                                               list(self.linear.parameters()), **optimiser_params)

        self.gpt.to(training_params["device"])
        self.distance.to(training_params["device"])
        self.linear.to(training_params["device"])
        self.training_params : dict = training_params

    def train(self, y_names : list[str],
              train_dataloader : DataLoader, 
              val_dataloader : DataLoader,
              test_dataloader : None | DataLoader = None):
        
        best_epoch : int = 0
        best_aggregate_metric : float = math.inf
        best_individual_metrics : list[float] = len(y_names) * [math.inf]

        # Training
        for epoch in range(self.training_params["epochs"]):
            loss_sum : float = 0
            for batch in train_dataloader:
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                self.optimiser.zero_grad()

                # get output from the model, given the inputs
                loss : torch.Tensor = self.train_step(batch, y_names)
                
                loss_sum += loss.item()
                # get gradients w.r.t to parameters
                loss.backward()

                # update parameters
                self.optimiser.step()

            loss_sum /= len(train_dataloader)

            # Evaluating
            aggregate_metric : float = 0
            individual_metrics : list[float] = [0] * len(y_names)
            
            instance_num : int = 0
            for batch in val_dataloader:
                instance_num += sum([len(sentence) for sentence in batch.word2tokenidxs])

                aggregate_metric_sen, individual_metrics_sen = self.eval_step(batch, y_names)

                print(aggregate_metric_sen)
                aggregate_metric += aggregate_metric_sen
                for i, metric_sen in enumerate(individual_metrics_sen):
                    individual_metrics[i] += metric_sen

            aggregate_metric /= instance_num
            for i in range(len(individual_metrics)):
                individual_metrics[i] /= instance_num

            log1 = 'epoch: {}, training loss: {:.2f}, val aggregate MSE: {:.2f},'.format(epoch, 
                                                                            loss_sum,
                                                                            aggregate_metric)
            
            log2 = " ".join(['val {} MAE: {:.2f}'.format(name, value) for name, value in zip(y_names, individual_metrics)])
            print("{} {}".format(log1, log2))

            if aggregate_metric < best_aggregate_metric:
                best_epoch = epoch
                best_aggregate_metric = aggregate_metric
                best_individual_metrics = individual_metrics

        # Report Best
        log1 : str = "Best epoch: {}, best val aggregate MSE: {:.2f},".format(best_epoch, best_aggregate_metric)
        log2 : str = ", ".join(['best val {} MAE: {:.2f}'.format(name, value) for name, value in zip(y_names, best_individual_metrics)])
        print("{} {}".format(log1, log2))

        # Reporting Test
        if test_dataloader is not None:
            pass #TODO

        # Saving

    def eval():
        raise NotImplementedError
    
    def calculate_distances(self, word_gpt_in : list[torch.Tensor], word_gpt_out : list[torch.Tensor]) -> list[torch.Tensor]:
        preds : list[torch.Tensor] = []

        for sentence_in, sentence_out in zip(word_gpt_in, word_gpt_out):
            embeddings = self.gpt.transformer.wte.weight
            vocab_size = embeddings.shape[0]

            o = sentence_out[:-1].unsqueeze(0).expand(vocab_size+1, *sentence_out[:-1].shape)

            i = embeddings.unsqueeze(1).expand(embeddings.shape[0], *sentence_out[:-1].shape)

            i = torch.cat((sentence_in[1:].unsqueeze(0), i), dim = 0)

            distances = self.distance(i, o)

            if self.invert:
                distances = 1 / distances

            distances = torch.nn.functional.softmax(distances, dim = 0)
            
            distances = distances[0].unsqueeze(-1)

            distances = self.linear(-torch.log(distances))

            preds.append(distances)

        return preds
    
    def train_step(self, batch : Batch, y_names : list[str]) -> torch.Tensor:

        input_ids : torch.Tensor = batch.input_ids.to(self.training_params["device"])

        gpt_in_embeds : torch.Tensor  
        gpt_out_embeds : torch.Tensor
        gpt_in_embeds, gpt_out_embeds = generate_embeds(input_ids, self.gpt)

        word_gpt_in : list[torch.Tensor] = []
        word_gpt_out : list[torch.Tensor] = []
        for sen_gpt_in, sen_gpt_out, sen_map in zip(gpt_in_embeds, gpt_out_embeds, batch.word2tokenidxs):
            sen_word_gpt_in : list[torch.Tensor] = [sen_gpt_in[0]]
            sen_word_gpt_out : list[torch.Tensor] = [sen_gpt_out[0]]
            for word in sen_map:
                sen_word_gpt_in.append(torch.mean(sen_gpt_in[word], dim = 0))
                sen_word_gpt_out.append(torch.mean(sen_gpt_out[word], dim = 0))
            
            word_gpt_in.append(torch.stack(sen_word_gpt_in))
            word_gpt_out.append(torch.stack(sen_word_gpt_out))

        preds : list[torch.Tensor] = self.calculate_distances(word_gpt_in, word_gpt_out)

        # TODO: should we normalise against distance to all words?

        return self.calculate_loss(preds, [torch.stack([sentence[name] for name in y_names], dim = -1) for sentence in batch])
    
    def eval_step(self, batch : Batch, y_names : list[str]) -> tuple[float, list[float]]:
        
        input_ids : torch.Tensor = batch.input_ids.to(self.training_params["device"])

        gpt_in_embeds : torch.Tensor  
        gpt_out_embeds : torch.Tensor
        gpt_in_embeds, gpt_out_embeds = generate_embeds(input_ids, self.gpt)

        word_gpt_in : list[torch.Tensor] = []
        word_gpt_out : list[torch.Tensor] = []
        for sen_gpt_in, sen_gpt_out, sen_map in zip(gpt_in_embeds, gpt_out_embeds, batch.word2tokenidxs):
            sen_word_gpt_in : list[torch.Tensor] = [sen_gpt_in[0]]
            sen_word_gpt_out : list[torch.Tensor] = [sen_gpt_out[0]]
            for word in sen_map:
                sen_word_gpt_in.append(torch.mean(sen_gpt_in[word], dim = 0))
                sen_word_gpt_out.append(torch.mean(sen_gpt_out[word], dim = 0))
            
            word_gpt_in.append(torch.stack(sen_word_gpt_in))
            word_gpt_out.append(torch.stack(sen_word_gpt_out))
        
        preds : list[torch.Tensor] = self.calculate_distances(word_gpt_in, word_gpt_out)

        word_gold : list[torch.Tensor] = [torch.stack([sentence[name] for name in y_names], dim = -1) for sentence in batch]

        def absolute_error(x : torch.Tensor, y : torch.Tensor) -> float:
            return (x-y).abs().sum().item()
        
        def squared_error(x : torch.Tensor, y : torch.Tensor) -> float:
            return (x-y).pow(2).sum().item()

        absolute_error_list : list[float] = [sum([absolute_error(x[:, n], y[:, n]) for x, y in zip(preds, word_gold)]) for n in range(len(y_names))]
        squared_error_list : list[float] = [sum([squared_error(x[:, n], y[:, n]) for x, y in zip(preds, word_gold)]) for n in range(len(y_names))]

        return sum(squared_error_list) / len(squared_error_list), absolute_error_list
    
    
    def calculate_loss(self, x : list[torch.Tensor], y : list[torch.Tensor]) -> torch.Tensor:
        loss : list[torch.Tensor] = []
        for x_sen, y_sen in zip(x, y):
            loss.append(self.loss(x_sen, y_sen))
        return sum(loss) / len(x)
    

import distances
import data

train_for = ["nFix"] #, "ffd", "gpt", "fixProp", "trt"]

# distances.BilinearDistance(768)
r = GPTDistanceRegression(len(train_for), distances.CosineSimilarity(), 
                          optimiser_params = {"lr" : 1e-10},
                          training_params = {"epochs" : 10, "device" : "cpu"})
r.train(train_for, data.dl, data.vdl)

#TODO Why does CosineSimilarity not work?
#TODO Should we normalise against distance to all other words? (like surprisal is)