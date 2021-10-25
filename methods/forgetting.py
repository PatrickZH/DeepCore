from CoresetMethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from .. import nets


# Acknowledgement to
# https://github.com/mtoneva/example_forgetting

class forgetting(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)

        # Initialize dictionary to save statistics for every example presentation
        self.example_stats = {}

    def get_hms(self, seconds):
        # Format time for printing purposes

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return h, m, s

    def train(self, model, model_optimizer, criterion, epoch):
        """ Train model for one epoch """

        train_loss = 0.
        correct = 0.
        total = 0.

        model.train()

        # Get permutation to shuffle trainset
        trainset_permutation_inds = np.random.permutation(np.arange(self.n_train))

        print('\n=> Training Epoch #%d' % epoch)
        batch_size = self.args.batch

        for batch_idx, batch_start_ind in enumerate(range(0, self.n_train, batch_size)):

            # Get trainset indices for batch
            batch_inds = trainset_permutation_inds[batch_start_ind:
                                                   batch_start_ind + batch_size]

            # Get batch inputs and targets, transform them appropriately
            transformed_trainset = []
            for ind in batch_inds:
                transformed_trainset.append(self.dst_train.__getitem__(ind)[0])
            inputs = torch.stack(transformed_trainset)
            targets = torch.LongTensor(
                np.array(self.dst_train.train_labels)[batch_inds].tolist())

            # Map to available device
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            model_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)

            # Update statistics and loss
            acc = predicted == targets
            for j, index in enumerate(batch_inds):

                # Get index in original dataset (not sorted by forgetting)
                index_in_original_dataset = self.train_indx[index]

                # Compute missclassification margin
                output_correct_class = outputs.data[j, targets[j].item()]
                sorted_output, _ = torch.sort(outputs.data[j, :])
                if acc[j]:
                    # Example classified correctly, highest incorrect class is 2nd largest output
                    output_highest_incorrect_class = sorted_output[-2]
                else:
                    # Example misclassified, highest incorrect class is max output
                    output_highest_incorrect_class = sorted_output[-1]
                margin = output_correct_class.item(
                ) - output_highest_incorrect_class.item()

                # Add the statistics of the current training example to dictionary
                index_stats = self.example_stats.get(index_in_original_dataset,
                                                     [[], [], []])
                index_stats[0].append(loss[j].item())
                index_stats[1].append(acc[j].sum().item())
                index_stats[2].append(margin)
                self.example_stats[index_in_original_dataset] = index_stats

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()
            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            loss.backward()
            model_optimizer.step()

            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item(),
                100. * correct.item() / total))

            # Add training accuracy to dict
            index_stats = self.example_stats.get('train', [[], []])
            index_stats[1].append(100. * correct.item() / float(total))
            self.example_stats['train'] = index_stats

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        model = nets.__dict__[self.args.model](self.args.channel, self.num_classes).to(self.args.device)
        criterion = nn.CrossEntropyLoss().to(self.args.device)
        criterion.__init__(reduce=False)

        # Setup optimizer
        model_optimizer = torch.optim.__dict__[self.args.optimizer](model.parameters(), lr=self.args.lr,
                                                                    momentum=self.args.momentum,
                                                                    weight_decay=self.args.weight_decay)

        best_acc = 0
        elapsed_time = 0

        for epoch in range(self.epochs):
            start_time = time.time()

            self.train(model, model_optimizer, criterion, epoch)
            # self.test(epoch, model)

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (self.get_hms(elapsed_time)))

    def select(self, **kwargs):
        self.run()
        _, unlearned_per_presentation, _, first_learned = self.compute_forgetting_statistics(self.example_stats,
                                                                                             self.epochs)
        ordered_examples, _ = self.sort_examples_by_forgetting(unlearned_per_presentation, first_learned, self.epochs)
        top_k_examples = ordered_examples[:self.coreset_size]

        return torch.utils.data.Subset(self.dst_train, top_k_examples), top_k_examples

    def sort_examples_by_forgetting(self, unlearned_per_presentation_all, first_learned_all, npresentations):
        """
        Sorts examples by number of forgetting counts during training, in ascending order
        If an example was never learned, it is assigned the maximum number of forgetting counts
        If multiple training runs used, sort examples by the sum of their forgetting counts over all runs

        unlearned_per_presentation_all: list of dictionaries, one per training run
        first_learned_all: list of dictionaries, one per training run
        npresentations: number of training epochs

        Returns 2 numpy arrays containing the sorted example ids and corresponding forgetting counts
        """

        # Initialize lists
        example_original_order = []
        example_stats = []

        for example_id in unlearned_per_presentation_all[0].keys():

            # Add current example to lists
            example_original_order.append(example_id)
            example_stats.append(0)

            # Iterate over all training runs to calculate the total forgetting count for current example
            for i in range(len(unlearned_per_presentation_all)):

                # Get all presentations when current example was forgotten during current training run
                stats = unlearned_per_presentation_all[i][example_id]

                # If example was never learned during current training run, add max forgetting counts
                if np.isnan(first_learned_all[i][example_id]):
                    example_stats[-1] += npresentations
                else:
                    example_stats[-1] += len(stats)

        print('Number of unforgettable examples: {}'.format(
            len(np.where(np.array(example_stats) == 0)[0])))
        return np.array(example_original_order)[np.argsort(
            example_stats)], np.sort(example_stats)

    def compute_forgetting_statistics(self, diag_stats, npresentations):
        """
        Calculates forgetting statistics per example

        diag_stats: dictionary created during training containing
                    loss, accuracy, and missclassification margin
                    per example presentation
        npresentations: number of training epochs

        Returns 4 dictionaries with statistics per example
        """

        presentations_needed_to_learn = {}
        unlearned_per_presentation = {}
        margins_per_presentation = {}
        first_learned = {}

        for example_id, example_stats in diag_stats.items():

            # Skip 'train' and 'test' keys of diag_stats
            if not isinstance(example_id, str):

                # Forgetting event is a transition in accuracy from 1 to 0
                presentation_acc = np.array(example_stats[1][:npresentations])
                transitions = presentation_acc[1:] - presentation_acc[:-1]

                # Find all presentations when forgetting occurs
                if len(np.where(transitions == -1)[0]) > 0:
                    unlearned_per_presentation[example_id] = np.where(
                        transitions == -1)[0] + 2
                else:
                    unlearned_per_presentation[example_id] = []

                # Find number of presentations needed to learn example,
                # e.g. last presentation when acc is 0
                if len(np.where(presentation_acc == 0)[0]) > 0:
                    presentations_needed_to_learn[example_id] = np.where(
                        presentation_acc == 0)[0][-1] + 1
                else:
                    presentations_needed_to_learn[example_id] = 0

                # Find the misclassication margin for each presentation of the example
                margins_per_presentation = np.array(
                    example_stats[2][:npresentations])

                # Find the presentation at which the example was first learned,
                # e.g. first presentation when acc is 1
                if len(np.where(presentation_acc == 1)[0]) > 0:
                    first_learned[example_id] = np.where(
                        presentation_acc == 1)[0][0]
                else:
                    first_learned[example_id] = np.nan

        return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned
