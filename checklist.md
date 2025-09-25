
## Assignment Checklist

### Setup & Environment
- [x] Run `bash setup.sh` successfully
- [x] Verify model weights downloaded (stories42M.pt)

### Core Implementation
- [x] **llama.py**
- [x] **rope.py**
- [x] **optimizer.py**
- [x] **classifier.py**
- [ ] **lora.py**

### Testing & Validation
- [x] Pass `python sanity_check.py`
- [x] Pass `python optimizer_test.py` 
- [x] Pass `python rope_test.py` 
- [x] Generate coherent text with `python run_llama.py --option generate`
- [ ] Complete SST zero-shot prompting
- [ ] Complete CFIMDB zero-shot prompting  
- [ ] Complete SST fine-tuning
- [ ] Complete CFIMDB fine-tuning
- [ ] Complete SST LoRA fine-tuning
- [ ] Complete CFIMDB LoRA fine-tuning

### Advanced Features (Optional - A+)
- [ ] Other advanced techniques (see A+ options above)
- [ ] Write 1-2 page report documenting improvements

### Submission Preparation
- [ ] Generate all required output files
- [ ] Validate submission with `python3 prepare_submit.py`
- [ ] Verify file size < 10MB
- [ ] Create zip file with proper ANDREWID structure
- [ ] Include optional report and feedback files
- [ ] Final submission check before Canvas upload
  
