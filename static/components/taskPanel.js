class TaskPanel extends HTMLElement {
    static get observedAttributes() {
        return ['title', 'description', 'tags', 'link']
    }
    constructor() {
        super()
    }

    connectedCallback() {
        this.innerHTML = `
            <link rel="stylesheet" href="/components/taskPanel.css" />
            <div class="panel">
                <a href="" id="taskLink"><h1 id="title"></h1></a>
                <h3 id="desc" style="margin"></h3>
                <div id="tagHolder" class="tagHolder">
                    <h3>Tags: </h3>
                </div>
            </div>
        `
        this.titleElem = this.querySelector("#title")
        this.taskLinkElem = this.querySelector("#taskLink")
        this.descriptionElem = this.querySelector("#desc")
        this.tagHolderElem = this.querySelector("#tagHolder")
    }

    attributeChangedCallback(attrName, oldVal, newVal) {
        if (attrName == "title") {
            this.titleElem.innerHTML = newVal;
        }
        else if (attrName == "description") {
            this.descriptionElem.innerHTML = newVal;
        }
        else if (attrName == "tags") {
            newVal = newVal.split(",")
            newVal.forEach(element => {
                let tag = document.createElement("h4")
                tag.style.borderRadius = "5px"
                tag.style.marginLeft = "10px"
                tag.style.padding = "2px 5px"
                tag.style.color = "#252526"
                tag.style.backgroundColor = "yellow"
                tag.innerHTML = element
                this.tagHolderElem.appendChild(tag)
            });
        }
        else if (attrName == "link") {
            this.taskLinkElem.setAttribute("href", newVal)
        }
    }
}
customElements.define("task-panel", TaskPanel)