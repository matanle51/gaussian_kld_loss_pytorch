# Define univariate kl loss
class UnivariateKLD(torch.nn.Module):
    def __init__(self, reduction):
        super(UnivariateKLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu1, mu2, logvar_1, logvar_2):
        mu1, mu2 = mu1.type(dtype=torch.float64), mu2.type(dtype=torch.float64)
        sigma_1 = logvar_1.exp().type(dtype=torch.float64)  # sigma_1 := sigma_nat^2
        sigma_2 = logvar_2.exp().type(dtype=torch.float64)  # sigma_2 := sigma_adv^2

        # log(sqrt(sigma2)/sqrt(sigma1))
        term_1 = (sigma_2.sqrt() / sigma_1.sqrt()).log()

        # (sigma_1 + (mu1-mu2)^2)/(2*sigma_2)
        term_2 = (sigma_1 + (mu1 - mu2).pow(2))/(2*sigma_2)

        # Calc kl divergence on entire batch
        kl = term_1 + term_2 - 0.5

        # Calculate mean kl_d loss
        if self.reduction == 'mean':
            kl_agg = torch.mean(kl)
        elif self.reduction == 'sum':
            kl_agg = torch.sum(kl)
        else:
            raise NotImplementedError(f'Reduction type not implemented: {self.reduction}')

        return kl_agg
