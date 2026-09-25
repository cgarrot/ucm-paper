"""SIW policy adapter (audit bug 5): ModelPolicy over the SIW tensorizer.

Order randomization reuses the V0 generic transform — SIW observations share
the canonical format ({action,arg} candidates, {subj,pred,obj} relations,
{id,type,attrs} entities, {predicate,args} goals), so randomize_obs_order is
format-compatible; the id-mapping covers SIW refs (views/widgets/agent).
"""

from __future__ import annotations

import mlx.core as mx

from ucm.eval.rollout import Action, ModelPolicy
from ucm.v1.tensorize_siw import collate_siw, tensorize_siw_obs


class SIWModelPolicy(ModelPolicy):
    def logits(self, obs: dict) -> mx.array:
        # NOTE (audit B1): order randomization happens ONCE, in the parent
        # __call__ — do NOT re-shuffle here (double shuffle broke the
        # action↔candidate mapping; T8-class bug, regression-tested).
        ex = tensorize_siw_obs(obs)
        ex["labels"] = None
        batch = collate_siw([ex])
        out = self.model(batch)[0]
        if self.validity_mask:
            # SIW preconditions are env-specific; the declared privileged arm
            # is wired when the SIW validity checker lands (freeze unchanged)
            raise NotImplementedError("SIW validity mask: pending SIW valid_actions")
        mx.eval(out)
        return out
