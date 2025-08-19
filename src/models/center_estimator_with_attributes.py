import torch

from models.center_estimator import CenterEstimator

from utils.utils import get_log_and_inverse_fn

class CenterAttributeEstimator(CenterEstimator):
    def __init__(self, args=dict(), is_learnable=True):
        super(CenterAttributeEstimator, self).__init__(args, is_learnable=is_learnable)

        self.attributes = args['attributes']

        assert type(self.attributes) is dict, "CenterAttributeEstimator expects 'attributes' setting as a dictionary with keys representing output prediction name and values representing input data index"

        # get max index requested as attribute to ensure correct number of fields will be present
        self.max_attributes = max([attr['input_start']+max(1,attr['input_end']) for _, attr in self.attributes.items()])
        
        ################################################################
        # Prepare log and inverse log functions
       
        for out_key, attr in self.attributes.items():
            if attr.get('use_log'):
                attr['log_fn'], attr['inverse_log_fn'] = get_log_and_inverse_fn(attr.get('use_log_base'))
        
        # make sure to disable instance maske in parent since we use  here
        assert not args.get('suppression_by_mask'), "ERROR: Cannot use suppression_by_mask=True with CenterAttributeEstimator"


    def init_output(self, num_vector_fields=1):
        super(CenterAttributeEstimator, self).init_output(num_vector_fields)

        REQUIRED_VECTOR_FIELDS = 3 + self.max_attributes
        
        assert self.num_vector_fields >= REQUIRED_VECTOR_FIELDS

    def forward(self, input, ignore_gt=False, **gt):
        ret = super(CenterAttributeEstimator, self).forward(input, ignore_gt, **gt)

        # use input from parent forward (in case this is modified)
        input = ret['output']
        center_pred = ret['center_pred']

        # first 3 values are sin,cos and R so skip them
        INPUT_OFFEST = 3

        predictions = input[:, INPUT_OFFEST:self.num_vector_fields]

        batch_size = center_pred.shape[0]
        num_pred = center_pred.shape[1]

        pred_attributes = dict()
        
        for out_key, attr in self.attributes.items():
            input_start = attr['input_start']
            input_end = attr.get('input_end')

            pred_attr_vals = predictions[:, input_start:input_end] if input_end is not None else predictions[:, input_start:]
            DIMS = pred_attr_vals.shape[1]

            prediction_attrib = torch.zeros((batch_size,num_pred,DIMS))

            for b in range(batch_size):
                # for every predicted center point find its center
                for i, pred in enumerate(center_pred[b]):
                    if pred[0] != 0:
                        x,y = pred[1:3]
                        prediction_attrib[b,i,:] = pred_attr_vals[b,:,int(y), int(x)]

            if attr.get('use_log'):
                inverse_log_fn = attr['inverse_log_fn'] 
                inverse_log_dims = attr['use_log_channels'] if 'use_log_channels' in attr else range(DIMS)

                for i in inverse_log_dims:
                    prediction_attrib[:,:,i] = inverse_log_fn(prediction_attrib[:,:,i])

            # add shape coef to list of returned values
            pred_attributes[out_key] = prediction_attrib

        if 'pred_attributes' not in ret:
            ret['pred_attributes'] = dict()

        ret['pred_attributes'].update(pred_attributes)

        return ret