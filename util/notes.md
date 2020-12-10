# callbacks
create a callback that plots for each meniscus 
- self attention
- cross attention   

## how to:
1) take 3 reference points

```
_____________________
|                   |
| x       x       x | <- 0.5 * boxHeight
|___________________|

  |       |       |
 0.125   0.5     0.825       * boxWidth
```

1) plot attention for those points  
=> repeat for target and ground truth   
=> repeat for best and worst case 