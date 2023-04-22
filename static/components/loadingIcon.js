class LoadingIcon extends HTMLElement{
  constructor(){
    super()
  }
  connectedCallback(){
    this.innerHTML = `
      <link rel="stylesheet" href="/components/loadingIcon.css"/>
      <div class="loader"></div>
    `

  }
}
customElements.define("loading-icon",LoadingIcon)