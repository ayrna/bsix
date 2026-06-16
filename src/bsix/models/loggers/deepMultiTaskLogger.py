import logging
import math

class DeepMultiTaskLogger():

    """
    Logger for model training and evaluation.
    """
        
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.history = {}

    def logMessage(self,message):
        self.logger.info(message)

    def print_progress_bar(self, step, max_steps, loss = None, validation_loss = None, ci = None, ci_valid = None, bar_length = 25, char = '*'):
        progress_length = int(bar_length * step / max_steps)
        progress_bar = [char] * (progress_length) + [' '] * (bar_length - progress_length)
        space_padding = int(math.log10(max_steps))
        if step > 0:
            space_padding -= int(math.log10(step))
        space_padding = ''.join([' '] * space_padding)
        message = "Training step %d/%d %s|" % (step, max_steps, space_padding) + ''.join(progress_bar) + "|"
        if loss:
            message += f" - loss: {loss}"
        if ci != None:
            message += f" - ci: {ci}"
        if validation_loss:
            message += f" - validation_loss: {validation_loss}"
        if ci_valid != None:
            message += f" - validation_ci: {ci_valid}"

        self.logger.info(message)

    def logValue(self, key, value, step):
        pass

    def shutdown(self):
        logging.shutdown()