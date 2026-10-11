events ─► 4 time surfaces ─► VSA encoder ─► cost volume ─► flow
                                   │                         │
                                   │                  ego-motion fit
                                   │                         │
                                   ▼                         ▼
                           appearance X (F0)    residual velocity + motion channels
                                   └──────────► CNN ◄────────┘
                                                 │  moving / static per pixel
                                                 ▼
                                   grouping ─► one colour per object