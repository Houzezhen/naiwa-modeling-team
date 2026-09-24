const pptxgen=require('./node_modules/pptxgenjs');
(async()=>{const p=new pptxgen(); p.layout='LAYOUT_WIDE'; const s=p.addSlide(); s.addText('test',{x:1,y:1,w:3,h:1}); await p.writeFile({fileName:'figures/test_simple.pptx'});})();
